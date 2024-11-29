import cv2
from . import scanLanes
def make_rectangles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert roi into gray
    Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
    Canny=cv2.Canny(Blur,10,50) #apply canny to roi
    #Find my contours
    contours =cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    print(len(contours))
    #Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
    cntrRect = []
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    for i in contours:
            epsilon = 0.05*cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,epsilon,True)
            if len(approx) == 4:
                cv2.drawContours(frame,cntrRect,-1,(0,255,0),2)
                cv2.drawContours(frame,cntrRect,-1,(45,67,23),-1)
                # cv2.imshow('Roi Rect ONLY',frame)
                cntrRect.append(approx)
                
            # Used to flatted the array containing
            # the co-ordinates of the vertices.
            n = approx.ravel() 
            i = 0
            
            for j in n :
                if(i % 2 == 0):
                    x = n[i]
                    y = n[i + 1]
        
                    # String containing the co-ordinates.
                    string = str(x) + " " + str(y) 
        
                    if(i == 0):
                        # text on topmost co-ordinate.
                        cv2.putText(frame, "Arrow tip", (x, y),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0)) 
                    else:
                        # text on remaining co-ordinates.
                        cv2.putText(frame, string, (x, y), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0)) 
                i = i + 1
    return frame, cntrRect