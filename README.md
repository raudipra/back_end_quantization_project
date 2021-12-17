# Backend Quantization Project
Part of: [Quantization Module](https://github.com/rluthfan/pytorch-quantization) and [Front End Quantization Project](https://github.com/raudipra/front_end_quantization_project)

This repository contains backend server codes for quantization project. The backend server acts as the middleware between front end and ONNX runtime server. Passing the image to ONNX, parse the response, return the prediction and inference time to front end.

## Setup

- `pip install -r requirements.txt`
- Edit the URL variables inside app.py correspondingly.

## How to Run
- `python3 app.py`