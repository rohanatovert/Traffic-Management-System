# def change_roi(x):
    #     print(x)
    # cv2.namedWindow('tracker')
    # cv2.createTrackbar("ROI:1","tracker",0, image.shape[1], change_roi)
    # cv2.createTrackbar("ROI:1","tracker",0, image.shape[0], change_roi)

import edgeDetection
import extractBG
# import shapeDetection
import roi
import completeLines
import markSections
import detectObjects
import cv2
import numpy as np
import scanLanes
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")

args = vars(ap.parse_args())

# file = r"C:\Users\rohan\Downloads\oVERT\Prod\Videos\cctv.mp4" #use this
file = args["input"] #use this
# file = r"C:\Users\rohan\Downloads\oVERT\Test\Videos\Traffic - 112541.mp4" #testing
frame = extractBG.extract(file)
cap = cv2.VideoCapture(file)
ret, original_frame = cap.read()
lane_image = np.copy(original_frame)
cv2.namedWindow('Tracker',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Tracker',1000,300)
def change_roi(x):
    print(x)

# cv2.createTrackbar("ROI:1","tracker",0, original_frame.shape[0], change_roi)


cv2.createTrackbar("Blur", "Tracker",1, 1, change_roi)
cv2.createTrackbar("Threshold 1","Tracker",150, 500, change_roi)
cv2.createTrackbar("Threshold 2","Tracker",200, 500, change_roi)
cv2.createTrackbar("Rho","Tracker",1, 10, change_roi)
cv2.createTrackbar("Theta","Tracker",25, 180, change_roi)
cv2.createTrackbar("LThreshold","Tracker",10, 100, change_roi)
cv2.createTrackbar("MinLength","Tracker",10, 300, change_roi)
cv2.createTrackbar("MaxLength","Tracker",250, 300, change_roi)
cv2.createTrackbar("X1", "Tracker",400, 1000, change_roi)
cv2.createTrackbar("Y1", "Tracker",300, original_frame.shape[0], change_roi)
cv2.createTrackbar("X2", "Tracker",800, 1200, change_roi)
cv2.createTrackbar("Y2", "Tracker",300, original_frame.shape[0], change_roi)
cv2.createTrackbar("X3", "Tracker",1200, 2000, change_roi)
cv2.createTrackbar("Y3", "Tracker",original_frame.shape[0], original_frame.shape[0], change_roi)
cv2.createTrackbar("X4", "Tracker",150, 1200, change_roi)
cv2.createTrackbar("Y4", "Tracker",original_frame.shape[0], original_frame.shape[0], change_roi)

while cap.isOpened():
    ret, original_frame = cap.read()
    gaussian_blur = cv2.getTrackbarPos("Blur",'Tracker')
    t1 = cv2.getTrackbarPos("Threshold 1",'Tracker')
    t2 = cv2.getTrackbarPos("Threshold 2",'Tracker')
    if cv2.getTrackbarPos("Rho",'Tracker')>0:
        rho = cv2.getTrackbarPos("Rho",'Tracker')
    if cv2.getTrackbarPos("Theta",'Tracker')>0:
        theta = np.pi/cv2.getTrackbarPos("Theta",'Tracker')
    threshold = cv2.getTrackbarPos("LThreshold",'Tracker')
    minLength = cv2.getTrackbarPos("MinLength",'Tracker')
    maxLength = cv2.getTrackbarPos("MaxLength",'Tracker')
    x1 = cv2.getTrackbarPos("X1",'Tracker')
    y1 = cv2.getTrackbarPos("Y1",'Tracker')
    x2 = cv2.getTrackbarPos("X2",'Tracker')
    y2 = cv2.getTrackbarPos("Y2",'Tracker')
    x3 = cv2.getTrackbarPos("X3",'Tracker')
    y3 = cv2.getTrackbarPos("Y3",'Tracker')
    x4 = cv2.getTrackbarPos("X4",'Tracker')
    y4 = cv2.getTrackbarPos("Y4",'Tracker')

    if gaussian_blur==0:
        do_blur = "no_blur"
    else:
        do_blur = "blur"
    
    nframe = roi.crop(frame,x1,y1,x2,y2,x3,y3,x4,y4)
    nframe = edgeDetection.detect(nframe,do_blur,t1,t2)
    cv2.imshow("edges",nframe)
    lines = cv2.HoughLinesP(nframe, rho, theta, threshold, None, minLength, maxLength)
    nframe, contours = completeLines.make_lines(lane_image, lines, x1,y1,x2,y2,x3,y3,x4,y4)
    if ret is False:
        print('no video')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
       
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    lane_image = np.copy(original_frame)
    # detectObjects.count(lane_image)
    # scanLanes.is_car_there(lane_image,contours)
    
    result_frame = cv2.addWeighted(lane_image, 0.8, nframe, 1, 1)
    cv2.imshow("Result",result_frame)

while True:
    ret, original_frame = cap.read()
    lane_image = np.copy(original_frame)
    if ret is False:
        break
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    scanLanes.is_car_there(lane_image,contours)
    result_frame = cv2.addWeighted(lane_image, 0.8, nframe, 1, 1)
    cv2.imshow("Result",result_frame)
cap.release()
cv2.destroyAllWindows()
