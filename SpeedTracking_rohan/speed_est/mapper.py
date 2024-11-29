import numpy as np
from typing import List

ROAD_WIDTH=35
ROAD_HEIGHT=210
SIDE_WALK_WIDTH=5

class PixelMapper:
    """
    To perform real world rectification with the help of homography matrix calculated using SVD
    for converting points from pixel plane to real world plane using known road measurements.

    Reference
    ---------
    ImmenselyHappy (https://math.stackexchange.com/users/688856/immenselyhappy), 
    Calculate Homography with and without SVD, URL (version: 2020-12-07): https://math.stackexchange.com/q/3511513
    """
    def __init__(self,pixel_plane,width=ROAD_WIDTH,height=ROAD_HEIGHT,side_width=SIDE_WALK_WIDTH) -> None:
        px1,px2,px3,px4=pixel_plane
        x_1=[px1[0],side_width]
        y_1=[px1[1],0]
        x_2=[px2[0],width+side_width]
        y_2=[px2[1],0]
        x_3=[px3[0],side_width]
        y_3=[px3[1],height]
        x_4=[px4[0],width+side_width]
        y_4=[px4[1],height]
        P = np.array([
            [-x_1[0], -y_1[0], -1, 0, 0, 0, x_1[0]*x_1[1], y_1[0]*x_1[1], x_1[1]],
            [0, 0, 0, -x_1[0], -y_1[0], -1, x_1[0]*y_1[1], y_1[0]*y_1[1], y_1[1]],
            [-x_2[0], -y_2[0], -1, 0, 0, 0, x_2[0]*x_2[1], y_2[0]*x_2[1], x_2[1]],
            [0, 0, 0, -x_2[0], -y_2[0], -1, x_2[0]*y_2[1], y_2[0]*y_2[1], y_2[1]],
            [-x_3[0], -y_3[0], -1, 0, 0, 0, x_3[0]*x_3[1], y_3[0]*x_3[1], x_3[1]],
            [0, 0, 0, -x_3[0], -y_3[0], -1, x_3[0]*y_3[1], y_3[0]*y_3[1], y_3[1]],
            [-x_4[0], -y_4[0], -1, 0, 0, 0, x_4[0]*x_4[1], y_4[0]*x_4[1], x_4[1]],
            [0, 0, 0, -x_4[0], -y_4[0], -1, x_4[0]*y_4[1], y_4[0]*y_4[1], y_4[1]],
            ])
        [U,S,Vt]=np.linalg.svd(P)
        self.homography=Vt[-1].reshape(3,3)

    def transform(self,pxpy)->List:
        pxpy.append(1)
        xylambda = (self.homography @ np.array(pxpy).transpose())
        xy=(xylambda/xylambda[-1]).tolist()
        xy.pop()
        return xy