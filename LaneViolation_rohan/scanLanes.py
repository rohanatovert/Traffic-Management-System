import cv2
import numpy as np
from . import detectObjects

def is_car_there(frame,contours,cropimage, plateimage,image_files):
    
    coords = []
    for c in contours:
        # print(cv2.contourArea(c))
        if 10<cv2.contourArea(c)<90000:
            x, y, w, h = cv2.boundingRect(c)
            coords.append(x)
    VIOLATION = min(coords)
    for c in contours:
        # print(f"AREA of {c}",cv2.contourArea(c))
        if 10<cv2.contourArea(c)<90000:
            x, y, w, h = cv2.boundingRect(c)
            # print(x, y, w, h)
            
            new_frame =  frame[y:y+h,x:x+w]
            # cv2.imshow("new_frame", new_frame)
            # cv2.waitKey(0)
            if x==min(coords):
                lane = "Lane 1"
            elif x==max(coords):
                lane = "Lane 3"
            else:
                lane = "Lane 2"

            if x==VIOLATION:
                detectObjects.detect(new_frame,True,lane,cropimage, plateimage,image_files)
                # cv2.putText(new_frame,"Bike Lane",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            else:
                detectObjects.detect(new_frame,False,lane,cropimage, plateimage,image_files)
            # Lane1 = x
            # Lane2 = x
            # cv2.imshow(f"{x}x{y}", new_frame)
            
