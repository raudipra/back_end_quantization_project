import os
import json
import time

from flask import Flask, request, Response
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image
import numpy as np
import requests

import assets.onnx_ml_pb2 as onnx_ml_pb2
import assets.predict_pb2 as predict_pb2

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

URL = {
    32: "http://35.223.125.89",
    8: "http://34.67.3.238",
    6: "http://34.133.79.67",
    4: "http://34.136.59.82"
}
image_width = 112
image_height = 112
CLASS_NAMES = ["tench", "English springer", "cassette player", "chain saw", 
                "church", "French horn", "garbage truck", "gas pump", 
                "golf ball", "parachute"]

def predict(image, model_type):
    img = image.resize((image_width, image_height), Image.BILINEAR)

    # Preprocess and normalize the image
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        
    # Create request message to be sent to the ORT server
    input_tensor = onnx_ml_pb2.TensorProto()
    input_tensor.dims.extend(norm_img_data.shape)
    input_tensor.data_type = 1
    input_tensor.raw_data = norm_img_data.tobytes()

    request_message = predict_pb2.PredictRequest()

    # For your model, the inputs name should be something else customized by yourself. Use Netron to find out the input name.
    request_message.inputs["input"].data_type = input_tensor.data_type
    request_message.inputs["input"].dims.extend(input_tensor.dims)
    request_message.inputs["input"].raw_data = input_tensor.raw_data

    content_type_headers = ['application/x-protobuf', 'application/octet-stream', 'application/vnd.google.protobuf']

    for h in content_type_headers:
        request_headers = {
            'Content-Type': h,
            'Accept': 'application/x-protobuf'
        }
        
    # Inference run using ORT server
    # Change the number 9001 to the appropriate port number if you had changed it during ORT Server docker instantiation

    PORT_NUMBER = 9001 # Change appropriately if needed based on any changes when invoking the server in the pre-requisites
    inference_url = URL[model_type] + ":" + str(PORT_NUMBER) + "/v1/models/default/versions/1:predict"
    response = requests.post(inference_url, headers=request_headers, data=request_message.SerializeToString())
    
    return response

def parse_response(response):
    # Parse response message

    response_message = predict_pb2.PredictResponse()
    response_message.ParseFromString(response.content)

    predictions = np.frombuffer(response_message.outputs['output'].raw_data, dtype=np.float32)
    return CLASS_NAMES[np.argmax(predictions)]

@app.route('/classify', methods=['POST'])
def classify():
    res = []
    model_type = int(request.form['model_type'])
    for f in request.files.getlist('images'):
        img = Image.open(f)
        start_time = int(time.time() * 1000)
        raw_prediction = predict(img, model_type)
        prediction = parse_response(raw_prediction)
        inference_time = int(time.time() * 1000) - start_time
        res.append({
            'label': prediction,
            'inference_time': inference_time
        })

    rsp = Response(json.dumps(res), status=200, content_type="application/json")
    return rsp

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001', debug=True)