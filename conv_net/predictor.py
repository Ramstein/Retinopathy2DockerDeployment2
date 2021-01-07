"""
This is the file that implements a flask server to do inferences. It's the file that you will modify to
implement the scoring for your own algorithm.
"""

from __future__ import absolute_import
from __future__ import print_function

import json
import multiprocessing
import os
from asyncio.log import logger
from collections import defaultdict
from os import path, makedirs

import boto3
import flask
import torch
from botocore.exceptions import ClientError
from catalyst.utils import load_checkpoint, unpack_checkpoint
from flask_jwt_extended.exceptions import NoAuthorizationError
from pandas import DataFrame
from pytorch_toolbelt.utils.torch_utils import to_numpy
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from conv_net.Retinopathy2.retinopathy.augmentations import get_test_transform
from conv_net.Retinopathy2.retinopathy.dataset import RetinopathyDataset
from conv_net.Retinopathy2.retinopathy.dataset import get_class_names
from conv_net.Retinopathy2.retinopathy.factory import get_model
from conv_net.Retinopathy2.retinopathy.inference import ApplySoftmaxToLogits, FlipLRMultiheadTTA, Flip4MultiheadTTA, \
    MultiscaleFlipLRMultiheadTTA
from conv_net.Retinopathy2.retinopathy.train_utils import report_checkpoint

'''Not Changing variables'''
data_dir = '/opt/ml/input/data'
bucket = "diabetic-retinopathy-data-from-radiology"
model_dir = '/opt/ml/model'
checkpoint_fname = 'model.pth'
num_workers = multiprocessing.cpu_count()
need_features = True
tta = None
apply_softmax = True

params = {}
CLASS_NAMES = []


def download_from_s3(region='us-east-1', bucket="diabetic-retinopathy-data-from-radiology", s3_filename='test.png',
                     local_path="/opt/ml/input/data"):
    if not path.exists(local_path):
        makedirs(local_path, mode=0o755, exist_ok=True)
    s3_client = boto3.client('s3', region_name=region)
    try:
        s3_client.download_file(bucket, Key=s3_filename, Filename=local_path)
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            logger.info(f"The object s3://{bucket}/{s3_filename} in {region} does not exist.")
        else:
            raise


def image_with_name_in_dir(dirname, image_id):
    for ext in ['png', 'jpg', 'jpeg', 'tif']:
        image_path = path.join(dirname, f'{image_id}.{ext}')
        if path.isfile(image_path):
            return image_path
    raise FileNotFoundError(image_path)


def run_image_preprocessing(
        params,
        image_df: DataFrame,
        image_paths=None,
        preprocessing=None,
        image_size=None,
        crop_black=True,
        **kwargs) -> RetinopathyDataset:
    if image_paths is not None:
        if preprocessing is None:
            preprocessing = params.get('preprocessing', None)

        if image_size is None:
            image_size = params.get('image_size', 1024)
            image_size = (image_size, image_size)

        if 'diagnosis' in image_df:
            targets = image_df['diagnosis'].values
        else:
            targets = None

        return RetinopathyDataset(image_paths, targets, get_test_transform(image_size,
                                                                           preprocessing=preprocessing,
                                                                           crop_black=crop_black))


def model_fn(model_dir):
    model_path = path.join(model_dir, checkpoint_fname)  # '/opt/ml/model/model.pth'

    # already available in this method torch.load(model_path, map_location=lambda storage, loc: storage)
    checkpoint = load_checkpoint(model_path)
    params = checkpoint['checkpoint_data']['cmd_args']

    model_name = 'seresnext50d_gap'

    if model_name is None:
        model_name = params['model']

    coarse_grading = params.get('coarse', False)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    CLASS_NAMES = get_class_names(coarse_grading=coarse_grading)
    num_classes = len(CLASS_NAMES)
    model = get_model(model_name, pretrained=False, num_classes=num_classes)
    unpack_checkpoint(checkpoint, model=model)
    report_checkpoint(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = model.eval()

    if apply_softmax:
        model = nn.Sequential(model, ApplySoftmaxToLogits())

    if tta == 'flip' or tta == 'fliplr':
        model = FlipLRMultiheadTTA(model)

    if tta == 'flip4':
        model = Flip4MultiheadTTA(model)

    if tta == 'fliplr_ms':
        model = MultiscaleFlipLRMultiheadTTA(model)

    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=[id for id in range(torch.cuda.device_count())])

    return model


def input_fn(request_body, request_content_type='application/json'):
    image_name = []

    if request_content_type == 'application/json':
        input_object = json.loads(request_body)
        region = input_object['region']

        logger.info('Downloading the input diabetic retinopathy data.')
        for i in range(100):
            try:
                img = input_object[f'img{str(i)}']
                download_from_s3(region=region, bucket=bucket, s3_filename=img, local_path=data_dir)
                image_name.append(img)
            except KeyError as e:
                print(e)
                break

        image_df = DataFrame(image_name, columns=['id_code'])
        image_paths = image_df['id_code'].apply(lambda x: image_with_name_in_dir(data_dir, x))

        # Preprocessing the images
        dataset = run_image_preprocessing(
            params=params,
            apply_softmax=True,
            need_features=params['need_features'],
            image_df=image_df,
            image_paths=image_paths,
            batch_size=params['batch_size'],
            tta='fliplr',
            workers=num_workers,
            crop_black=True)

        return DataLoader(dataset, params['batch_size'],
                          pin_memory=True,
                          num_workers=num_workers)

    raise Exception(f'Requested unsupported ContentType in request_content_type {request_content_type}')


def predict_fn(input_object, model):
    predictions = defaultdict(list)

    for batch in tqdm(input_object):
        input = batch['image']
        if torch.cuda.is_available():
            input = input.cuda(non_blocking=True)
        outputs = model(input)

        predictions['image_id'].extend(batch['image_id'])
        if 'targets' in batch:
            predictions['diagnosis'].extend(to_numpy(batch['targets']).tolist())

        predictions['logits'].extend(to_numpy(outputs['logits']).tolist())
        predictions['regression'].extend(to_numpy(outputs['regression']).tolist())
        predictions['ordinal'].extend(to_numpy(outputs['ordinal']).tolist())
        if need_features:
            predictions['features'].extend(to_numpy(outputs['features']).tolist())

    del input_object
    return predictions


def output_fn(prediction, content_type='application/json'):
    """
    # Convert result to JSON
    """
    if content_type == 'application/json':
        return json.dumps(prediction), content_type
    else:
        raise Exception(f'Requested unsupported ContentType in Accept:{content_type}')


# def write_test_image(stream):
#     with open(TEST_IMG, "bw") as f:
#         chunk_size = 4096
#         while True:
#             chunk = stream.read(chunk_size)
#             if len(chunk) == 0:
#                 return
#             f.write(chunk)


# A singleton for holding the model. This simply loads the model and holds it.
# It has a InputPredictOutput function that does a prediction based on the model and the input data.

class ClassificationService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def IsVerifiedUser(cls, request):
        """Get the json data from flask.request."""
        if request.content_type == 'application/json':
            return True
            # if cls.json_data['request_type'] == 'inference':
            #     token_key = cls.json_data['token_key']
            #     # verify the token, is this person our user or not
            # else:
            #     pass
            #
            # try:
            #     img0 = cls.json_data['img0']
            #     if not img0:
            #         raise BadRequest()
            # except BadRequest:
            #     raise NoAuthorizationError(f'Missing img0 key in json data.')
        else:
            return False

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            cls.model = model_fn(model_dir=model_dir)
        return cls.model

    @classmethod
    def InputPredictOutput(cls, request_body, request_content_type='application/json'):
        """For the input, do the predictions and return them.
        Args:
            request_body (json request): containing region, and img0, img1 structure"
            request_content_type: 'application/json"""

        return output_fn(prediction=predict_fn(input_object=input_fn(request_body=request_body,
                                                                     request_content_type=request_content_type),
                                               model=cls.get_model()))


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ClassificationService.get_model() is not None  # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    print("cleaning test dir")
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            os.unlink(os.path.join(root, f))

    # write the request body to test file
    if ClassificationService.IsVerifiedUser(flask.request):  # verify the user with access type and token
        result = ClassificationService.InputPredictOutput(request_body=flask.request.get_json(force=True),
                                                          request_content_type='application/json')
    else:
        raise NoAuthorizationError('Invalid content-type. Must be application/json.')

    return flask.Response(response=result, status=200, mimetype='application/json')
