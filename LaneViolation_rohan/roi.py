import matplotlib.pyplot as plt
import cv2
import numpy as np
from . import extractBG
# file = r"C:\Users\rohan\Downloads\oVERT\Prod\Videos\cctv.mp4"
# image = extractBG.extract(file)
def crop(image,X1,Y1,X2,Y2,X3,Y3,X4,Y4):
    height = image.shape[0]
    # src = np.array([
    #     [(150,height),
    #     (1200,height),
    #     (800,300),
    #     (400,300)]
    #     ])
    src = np.array([
        [(X4,Y4),
        (X3,Y3),
        (X2,Y2),
        (X1,Y1)]
        ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask,src,255)
    masked_image = cv2.bitwise_and(image, mask)
    
    # plt.imshow(masked_image)
    # plt.show()
    return masked_image

# crop(image)
def front_to_top(img):
    height = img.shape[0]
    src = np.float32([
        [(500,200),
        (150,height),
        (1200,height),
        (700,200)]
        ])
    dst = np.float32([(100,0),
                    (100,720),
                    (1100,720),
                    (1100,0)
                    ])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    size = (1280, 720)
    frame = cv2.warpPerspective(img, M, size, flags = cv2.INTER_LINEAR)
    return frame