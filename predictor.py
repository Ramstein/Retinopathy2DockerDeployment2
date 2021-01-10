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
from flask_jwt_extended.exceptions import NoAuthorizationError

from inference import model_fn, input_fn, predict_fn, output_fn

'''Not Changing variables'''
data_dir = '/home/endpoint/data'
model_dir = '/home/model'
checkpoint_fname = 'model.pth'
model_name = 'seresnext50d_gap'
bucket = "diabetic-retinopathy-data-from-radiology"

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
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            cls.model = model_fn(model_dir=model_dir, model_name=model_name, checkpoint_fname=checkpoint_fname,
                                 apply_softmax=apply_softmax, tta=tta)
        return cls.model

    @classmethod
    def InputPredictOutput(cls, model, request_body, request_content_type='application/json'):
        """For the input, do the predictions and return them.
        Args:
            request_body (json request): containing region, and img0, img1 structure"
            request_content_type: 'application/json"""

        return output_fn(prediction=predict_fn(input_object=input_fn(request_body=request_body,
                                                                     request_content_type=request_content_type,
                                                                     data_dir=data_dir),
                                               model=model, need_features=need_features))


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ClassificationService.get_model() is not None  # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


# @app.route('/invocations', methods=['POST'])
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
            if ClassificationService.IsVerifiedUser(flask.request):  # verify the user with access type and token
                model = ClassificationService.get_model()
                result = ClassificationService.InputPredictOutput(model=model,
                                                                  request_body=flask.request.get_json(force=True),
                                                                  request_content_type='application/json')
            else:
                raise NoAuthorizationError('Invalid content-type. Must be application/json.')

            return flask.Response(response=result, status=200, mimetype='application/json')

            pred = predict(image_location, MODEL)[0]
            return render_template("index.html", prediction=pred, image_loc=image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    if not path.exists(data_dir):
        makedirs(data_dir, mode=0o755, exist_ok=True)

    health = ClassificationService.get_model() is not None  # You can insert a health check here
    status = 200 if health else 404
    print("status:", status)
    app.run(host="0.0.0.0", port='8080', debug=True)

    # return flask.Response(response='\n', status=status, mimetype='application/json')
