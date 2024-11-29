import cv2
import numpy as np

# oimage = cv2.imread(r"C:\Users\rohan\Downloads\oVERT\Prod\Images\preview.jpg")
# cv2.imshow("original",oimage)
def detect(oimage,do_blur,t1,t2): # ---> Remove KSIZE
    image = cv2.cvtColor(oimage, cv2.COLOR_BGR2GRAY)
    if do_blur=="blur":
        blur = cv2.GaussianBlur(image,(5,5),0) # -- Trackbar
    elif do_blur=="no_blur":
        blur = image
    # cv2.imshow("Image",image)
    # canny_image = cv2.Canny(blur,150,200)  # --> trackbar
    canny_image = cv2.Canny(blur,t1,t2)  # --> trackbar
    # cv2.imshow("Canny",canny_image)

    #REMOVE NOISE FOR LANE DETECTION

    #Erosion 
    kernel = np.ones((5,5), np.uint8)
    image = cv2.erode(canny_image, kernel , iterations = 1)
    # cv2.imshow("Erosion",image)

    #Dialation
    dialate = cv2.dilate(canny_image, kernel , iterations = 1)
    # cv2.imshow("Dialation",image)

    erode = cv2.erode(dialate, kernel , iterations = 1)
    # cv2.imshow("Erosion2",erode)

    # display = np.hstack((canny_image, dialate, erode))
    # cv2.imshow("Result",erode)
    # cv2.waitKey(0)
    return erode
