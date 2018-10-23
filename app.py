# -*- coding: utf-8 -*-
"""
Router for the endpoint of the image classifier

For a detailed implementation example, see: https://blog.keras.io/index.html

@author: AI team
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from main import prepare_image
import io
import os
from PIL import Image

from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras import models
import tensorflow as tf

app = Flask(__name__)
CORS(app)

#function to load keras model
classes = None
model = None
graph = None
def load_model():
    global classes
    global graph
    global model
    model = models.load_model('data/trained_models/trained_model.h5') 
    #identify the different classes
    classes = list(os.walk('data/image_dataset/val'))[0][1]
    graph = tf.get_default_graph()

@app.route('/classify', methods=['POST'])
def classify():
    #initialize the returned data dictionary
    data = {"success": False}
    
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image
            image = prepare_image(image)

            # classify the image
            with graph.as_default():
                results = model.predict(image)[0]
            data["predictions"] = []

            # loop over the results and add them to returned dictionary
            for i in range(len(classes)):
                r = {"label": classes[i], "probability": float(results[i])}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data as a JSON
    return jsonify(data)
    
if __name__ == '__main__':
    print('Loading classification model')
    load_model()
    app.run()