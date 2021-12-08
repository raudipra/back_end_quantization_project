import os
import json

from flask import Flask, request, Response
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

image_width = 300
image_height = 300
def predict(image_path):
    img = Image.open(image_path)
    img = img.resize((image_width, image_height), Image.BILINEAR)

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
    request_message.inputs["input.1"].data_type = input_tensor.data_type
    request_message.inputs["input.1"].dims.extend(input_tensor.dims)
    request_message.inputs["input.1"].raw_data = input_tensor.raw_data

    content_type_headers = ['application/x-protobuf', 'application/octet-stream', 'application/vnd.google.protobuf']

    for h in content_type_headers:
        request_headers = {
            'Content-Type': h,
            'Accept': 'application/x-protobuf'
        }
        
    # Inference run using ORT server
    # Change the number 9001 to the appropriate port number if you had changed it during ORT Server docker instantiation

    PORT_NUMBER = 9001 # Change appropriately if needed based on any changes when invoking the server in the pre-requisites
    inference_url = "http://127.0.0.1:" + str(PORT_NUMBER) + "/v1/models/default/versions/1:predict"
    response = requests.post(inference_url, headers=request_headers, data=request_message.SerializeToString())
    
    return response

def parse_response(response):
    # Parse response message

    response_message = predict_pb2.PredictResponse()
    response_message.ParseFromString(response.content)

    # For your model, the outputs names should be something else customized by yourself. Use Netron to find out the outputs names.
    boxes = np.frombuffer(response_message.outputs['boxes'].raw_data, dtype=np.float32)
    scores = np.frombuffer(response_message.outputs['scores'].raw_data, dtype=np.float32)

    print('Boxes shape:', response_message.outputs['boxes'].dims)
    print('Scores shape:', response_message.outputs['scores'].dims)
    
    boxes = torch.from_numpy(boxes.reshape(response_message.outputs['boxes'].dims))
    scores = torch.from_numpy(scores.reshape(response_message.outputs['scores'].dims))
    boxes.size(), scores.size()
    
    boxes = boxes[0]
    scores = scores[0]
    
    prob_threshold = 0.5
    nms_method = None
    iou_threshold = 0.45
    candidate_size = 200
    sigma = 0.5
    top_k = -1

    # this version of nms is slower on GPU, so we move data to CPU.
    boxes = boxes.to("cpu")
    scores = scores.to("cpu")
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, scores.size(1)):
        probs = scores[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.size(0) == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
        box_probs = box_utils.nms(box_probs, nms_method,
                                  score_threshold=prob_threshold,
                                  iou_threshold=iou_threshold,
                                  sigma=sigma,
                                  top_k=top_k,
                                  candidate_size=candidate_size)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.size(0))

    bboxes = torch.tensor([])
    labels = torch.tensor([])
    confidence = torch.tensor([])

    if picked_box_probs:
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= image_width
        picked_box_probs[:, 1] *= image_height
        picked_box_probs[:, 2] *= image_width
        picked_box_probs[:, 3] *= image_height

        bboxes = picked_box_probs[:, :4]
        labels = torch.tensor(picked_labels)
        confidence = picked_box_probs[:, 4]
        
    return bboxes, labels, confidence

@app.route('/classify', methods=['POST'])
def classify():
    res = []
    for f in request.files.getlist('images'):
        print(f)
        img = Image.open(f)
        print(img.size)
        f.save('./uploads/%s' % secure_filename(f.filename))
        res.append({
            'label': 'cats',
            'confidence': 0.54,
            'inference_time': 0.3
        })

    rsp = Response(json.dumps(res), status=200, content_type="application/json")
    return rsp

if __name__ == '__main__':
    if not os.path.exists('./uploads'):
        os.mkdir('./uploads')
    app.run(debug=True)