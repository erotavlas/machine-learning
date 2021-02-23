import os
import sys

from flask import Flask, request, Response, json
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model

import numpy as np

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'tmp'
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])

status = {}

try:
    model = load_model("cnn_2.ml")
    status["isready"] = True
    status["message"] = ""
except BaseException as error:
    status["isready"] = False
    status["message"] = "Error 3:"  + str(sys.exc_info()[0]) + str(error)


def load(filename):
    np_image = io.imread(filename)
    np_image = rgb2gray(np_image)/255
    np_image = resize(np_image, (128, 128, 1))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


@app.route("/classify", methods=['POST'])
def get_classification():
    try:
        if(status["isready"] == False):
            return Response(json.dumps({
                "code": 500,
                "name": 'Internal Server Error',
                "description": status["message"],
            }), status=500, mimetype='application/json')

        file = request.files['file']
        
        filename = secure_filename(file.filename)
    
        # check for supported file types
        extension = os.path.splitext(filename)[1].lstrip('.').lower()
        if extension in app.config['ALLOWED_EXTENSIONS']:

            img = os.path.join(app.config['UPLOAD_FOLDER'], filename)
          
            file.save(img)
           
            # process image 
            xray = load(img)

            # the predicted class
            prediction = model.predict_classes(xray)
            pred = int(prediction[0][0])

            # the probability of it being 1
            probability = (model.predict_proba(xray))[0][0]

            if pred == 0:
                probability = 1 - probability

            prob = float(probability)

            # cleanup
            os.remove(img)

            classes = {'normal': 0, 'pneumonia': 1}

            return json.dumps({"file": file.filename, "prediction": pred, 'probability': prob, 'classes': classes})
        else:
            raise Exception("File type unsupported")

    except BaseException as error:
        return Response(json.dumps({
            "code": 500,
            "name": 'Internal Server Error',
            "description": "Error 3:"  + str(sys.exc_info()[0]) + str(error),
        }), status=500, mimetype='application/json')    


if __name__ == "__main__":
    app.run(host='0.0.0.0')

