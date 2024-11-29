#Hough Tranform
import cv2
import numpy as np
from . import markSections
def make_lines(image,lines, X1, Y1, X2, Y2, X3, Y3, X4, Y4):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            line = line.astype(int)
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    height = image.shape[0]
    
    # cv2.line(line_image,(X4,Y4-50),(X3,Y3-50),(255,250,250),10)
    cv2.line(line_image,(X4,Y4-50),(X3,Y3-50),(255,250,250),10)
    cv2.line(line_image,(X2,Y2+10),(X1,Y1+10),(255,250,250),10)
    
    line_image = markSections.make_rectangles(line_image)
    # cv2.imshow("Result",line_image)
    # cv2.waitKey(0)
    return line_image

def averaged_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2),1)
        slope,intercept = parameters[0], parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
        
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_coordinates(image,left_fit_avg)
    right_line = make_coordinates(image,right_fit_avg)
    return np.array([left_line,right_line])

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(2/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

##merge lines that are near to each other based on distance
from numpy.polynomial import polynomial as P
def merge_lines(lines):
    clusters =  []
    idx = []
    total_lines = len(lines)
    if total_lines < 30:
        distance_threshold = 20
    elif total_lines <75:
        distance_threshold = 15
    elif total_lines<120:
        distance_threshold = 10
    else:
        distance_threshold = 7
    for i,line in enumerate(lines):
        x1,y1,x2,y2 = line.reshape(4)
        if [x1,y1,x2,y2] in idx:
            continue
        parameters = P.polyfit((x1, x2),(y1, y2), 1)
        slope = parameters[0]#(y2-y1)/(x2-x1+0.001)
        intercept = parameters[1]#((y2+y1) - slope *(x2+x1))/2
        a = -slope
        b = 1
        c = -intercept
        d = np.sqrt(a**2+b**2)
        cluster = [line]
    for d_line in lines[i+1:]:
        x,y,xo,yo= d_line
        mid_x = (x+xo)/2
        mid_y = (y+yo)/2
        distance = np.abs(a*mid_x+b*mid_y+c)/d
        if distance < distance_threshold:
            cluster.append(d_line)
            idx.append(d_line.tolist())
    clusters.append(np.array(cluster))
    merged_lines = [np.mean(cluster, axis=0) for cluster in clusters]
    return merged_lines