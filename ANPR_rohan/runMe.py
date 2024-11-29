import torch
import cv2
import easyocr

# cap = cv2.VideoCapture(r'C:\Users\rohan\Downloads\oVERT\Test\Videos\cars.mp4')
# # Model
# ret, frame = cap.read()
# # Image
# im = frame
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\rohan\Downloads\oVERT\yolov5-master-20221212T054236Z-001\yolov5-master\License_Plates\best.pt')

# results = model(im)
# labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
# print(labels, cord_thres)
# results.show()
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")

args = vars(ap.parse_args())

def recognize_text(img_path):
    '''loads an image and recognizes text.'''
    
    reader = easyocr.Reader(['en'])
    return reader.readtext(img_path)

# result = recognize_text(r"C:\Users\rohan\Downloads\oVERT\yolov5-master-20221212T054236Z-001\yolov5-master\runs\detect\exp33\crops\vehicle\cars3343.jpg")
result = recognize_text(args["input"])
print(result)
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     # Image
#     im = frame

#     # Inference
#     results = model(im)
#     print(results)
#     # results.pandas().xyxy[0]
#     # results.print()
#     results.show()
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
#python detect.py --weights C:\Users\rohan\Downloads\oVERT\yolov5-master-20221212T054236Z-001\yolov5-master\License_Plates\new_weights.py\best.pt --img 416 --conf 0.4 --source C:\Users\rohan\Downloads\oVERT\Test\Videos\cars.mp4 --save-crop --view-img
