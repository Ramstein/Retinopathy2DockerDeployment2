"""
This is the file that implements a flask server to do inferences. It's the file that you will modify to
implement the scoring for your own algorithm.
"""

from __future__ import absolute_import
from __future__ import print_function

import os
from os import path, makedirs

import flask
from flask import render_template
from flask import request

from S3Handler import S3Handler
from inference import model_fn, input_fn, predict_fn

'''Not Changing variables'''
region = 'us-east-1'
model_name = 'seresnext50d_gap'
model_bucket = 'diabetic-retinopathy'
checkpoint_fname = 'model.pth'
model_dir = '/home/model'

bucket = "diabetic-retinopathy-data-from-radiology"
data_dir = '/home/endpoint/data'

need_features = True
tta = None
apply_softmax = True


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
    def upload_to_s3(cls, channel, file, bucket, region=''):
        S3Handler.upload_to_s3(channel, file, bucket, region)

    @classmethod
    def download_from_s3(cls, region, bucket, s3_filename, local_path):
        S3Handler.download_from_s3(region, bucket, s3_filename, local_path)

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            cls.model = model_fn(model_dir=model_dir, model_name=model_name, checkpoint_fname=checkpoint_fname,
                                 apply_softmax=apply_softmax, tta=tta)
        return cls.model

    @classmethod
    def InputPredictOutput(cls, image_location, model):
        """For the input, do the predictions and return them.
        Args:"""
        input_object = input_fn(image_location, data_dir=data_dir)
        return predict_fn(input_object=input_object, model=model, need_features=need_features)
        # return output_fn(prediction=prediction)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ClassificationService.get_model() is not None  # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/', methods=['GET', 'POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    # print("cleaning test dir")
    # for root, dirs, files in os.walk(data_dir):
    #     for f in files:
    #         os.unlink(os.path.join(root, f))

    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(data_dir, image_file.filename)
            image_file.save(image_location)
            # write the request body to test file
            model = ClassificationService.get_model()
            result = ClassificationService.InputPredictOutput(image_location, model=model)  # result is a dict
            # result = {'image_id': "/home/endpoint/data/test.png",
            #           'logits': 65651,
            #           'regression': 4545,
            #           'ordinal': 98,
            #           'features': 'ghaf',
            #           }
            render_template("index.html", prediction=result['regrssion'], image_loc=image_file.filename)
            ClassificationService.upload_to_s3(channel="pretrained", file='se_resnext50_32x4d-a260b3a4.pth',
                                               bucket=bucket, region=region)

    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    if not path.exists(data_dir):
        makedirs(data_dir, exist_ok=True)

    if not path.exists(model_dir):
        makedirs(model_dir, exist_ok=True)

    if not path.isfile(path.join(model_dir, checkpoint_fname)):
        ClassificationService.download_from_s3(region=region, bucket=model_bucket,
                                               s3_filename='deployment/' + checkpoint_fname,
                                               local_path=path.join(model_dir, checkpoint_fname))

    health = ClassificationService.get_model() is not None  # You can insert a health check here
    status = 200 if health else 404
    print("status:", status)
    app.run(host="localhost", port=8080, debug=True)

    # return flask.Response(response='\n', status=status, mimetype='application/json')
