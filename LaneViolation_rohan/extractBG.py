import numpy as np
import cv2
import os

# video = cv2.VideoCapture(r"C:\Users\rohan\Downloads\oVERT\Prod\Videos\cctv.mp4")
def extract(filename):
    name = os.path.basename(filename).split(".")[0]+".jpg"
    video = cv2.VideoCapture(filename)
    FOI = video.get(cv2.CAP_PROP_FRAME_COUNT)*np.random.uniform(size=100) #30
    print(filename)
    frames = []
    for frame in FOI:
        video.set(cv2.CAP_PROP_FRAME_COUNT, frame)
        ret,frame = video.read()
        
        frames.append(frame)
        
    bgFrame = np.median(frames,axis=0).astype(dtype=np.uint8)
    cv2.imwrite(name,bgFrame)
    print("saving extracted BG")
    # cv2.imshow("BG",bgFrame)
    # cv2.waitKey(0)
    return bgFrame

