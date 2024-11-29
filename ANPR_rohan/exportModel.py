import torch
import torch.onnx
import torchvision
import torchvision.models as models
import sys
 
onnx_model_path = r"C:\Users\rohan\Downloads\oVERT\yolov5-master-20221212T054236Z-001\yolov5-master\License_Plates"

# https://pytorch.org/hub/pytorch_vision_densenet/
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\rohan\Downloads\oVERT\yolov5-master-20221212T054236Z-001\yolov5-master\License_Plates\best.pt')

# set the model to inference mode
model.eval()
 
# Create some sample input in the shape this model expects 
# This is needed because the convertion forward pass the network once 
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)