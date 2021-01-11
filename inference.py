from __future__ import absolute_import
from __future__ import print_function

import json
import multiprocessing
from collections import defaultdict
from os import path

import torch
from catalyst.utils import unpack_checkpoint, load_checkpoint
from pandas import DataFrame
from pytorch_toolbelt.utils.torch_utils import to_numpy
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Retinopathy2.retinopathy.augmentations import get_test_transform
from Retinopathy2.retinopathy.dataset import RetinopathyDataset, get_class_names
from Retinopathy2.retinopathy.factory import get_model
from Retinopathy2.retinopathy.inference import ApplySoftmaxToLogits, FlipLRMultiheadTTA, Flip4MultiheadTTA, \
    MultiscaleFlipLRMultiheadTTA
from Retinopathy2.retinopathy.train_utils import report_checkpoint

num_workers = multiprocessing.cpu_count()
params = {}
CLASS_NAMES = []


def image_with_name_in_dir(dirname, image_id):
    for ext in ['png', 'jpg', 'jpeg', 'tif']:
        if ext == str(image_id).split('.')[-1]:
            break
    if path.isfile(image_id):
        return image_id
    raise FileNotFoundError(image_id)


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


def model_fn(model_dir, model_name=None, checkpoint_fname='', apply_softmax=True, tta=None):
    model_path = path.join(model_dir, checkpoint_fname)  # '/home/model/model.pth'

    # already available in this method torch.load(model_path, map_location=lambda storage, loc: storage)

    checkpoint = load_checkpoint(model_path)
    params = checkpoint['checkpoint_data']['cmd_args']

    if model_name is None:
        print("Quitting, specify the model_name.")
        # model_name = params['model']
        return

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


def input_fn(image_location='', data_dir='', need_features=True):
    image_locations = []
    image_locations.append(image_location)
    # for i in range(13):  # 3 values for region, access, token
    #     try:
    #         img = input_object[f'img{str(i)}']
    #         image_name.append(img)
    #     except KeyError as e:
    #         print(e)

    image_df = DataFrame(image_locations, columns=['id_code'])
    image_paths = image_df['id_code'].apply(lambda x: image_with_name_in_dir(data_dir, x))

    # Preprocessing the images
    dataset = run_image_preprocessing(
        params=params,
        apply_softmax=True,
        need_features=need_features,
        image_df=image_df,
        image_paths=image_paths,
        batch_size=len(image_paths['id_code']),
        tta='fliplr',
        workers=num_workers,
        crop_black=True)

    return DataLoader(dataset, len(image_paths['id_code']),
                      pin_memory=True,
                      num_workers=num_workers)


def predict_fn(input_object, model, need_features):
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
