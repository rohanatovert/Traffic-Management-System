# python finalPro.py -i "C:\Users\rohan\Downloads\oVERT\Test\Videos\pexels-kelly-lacy.mp4"

import numpy as np
import cv2
import matplotlib.pyplot as plt
from . import scanLanes
import argparse
from . import extractBG
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
counter = 0

def get_road_map(source,stframe):
    file = source 
    print("Extracting BG...")
    frame = extractBG.extract(file)
    frame = cv2.resize(frame,(1080, 720))
    # cv2.imwrite("ExtractedBG.jpg",frame)
    return frame


def main(opt, image, bottom_left, top_left, top_right, bottom_right, stframe, cropimage, plateimage, image_files):
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--input", required=True,
    #     help="path to input video")

    # args = vars(ap.parse_args())

    # file = args["input"] #use this
    file = opt.source 
    # print("Extracting BG...")
    # frame = extractBG.extract(file)

    # image = cv2.imread(r"C:\Users\rohan\Downloads\oVERT\Github\preview2.jpg")
    

    # print('This image is:', type(image), 'with dimension:', image.shape)

    def grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def canny(img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(img, vertices):

        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def select_rgb_white_yellow(image):
        # white color mask
        lower = np.uint8([200, 200, 200])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(image, lower, upper)
        # yellow color mask
        lower = np.uint8([190, 190, 0])
        upper = np.uint8([255, 255, 255])
        yellow_mask = cv2.inRange(image, lower, upper)
        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked = cv2.bitwise_and(image, image, mask=mask)
        return masked


    # filtered_color = select_rgb_white_yellow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    filtered_color = image
    blurred = gaussian_blur(canny(filtered_color, 100, 150), 3)
    height, width = image.shape[:2]
    stframe.image(blurred, use_column_width=True)
    # cv2.imshow("blurred", blurred)
    # cv2.waitKey(0)
    # Create point matrix get coordinates of mouse click on image
    # point_matrix = np.zeros((4,2),np.int)
    # counter = 0
    # endloop = False
    # def mousePoints(event,x,y,flags,params):
    #     global counter, endloop
    #     # Left button mouse click event opencv
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         point_matrix[counter] = x,y
    #         counter = counter + 1
    #         print(counter)
            
    #     if event == cv2.EVENT_RBUTTONDOWN:
    #         endloop = True
    #     if counter>=4:
    #         endloop = True
        
    # while True:
    #     for x in range (0,4):
    #         cv2.circle(image,(point_matrix[x][0],point_matrix[x][1]),3,(0,255,0),cv2.FILLED)
    #     topLeft_x = point_matrix[0][0]
    #     topLeft_y = point_matrix[0][1]

    #     topRight_x = point_matrix[1][0]
    #     topRight_y = point_matrix[1][1]

    #     bottomLeft_x = point_matrix[3][0]
    #     bottomLeft_y = point_matrix[3][1]

    #     bottomRight_x = point_matrix[2][0]
    #     bottomRight_y = point_matrix[2][1]

    
    #     # Showing original image
    #     cv2.imshow("Original Image ", image)
    #     # Mouse click event on original image
    #     cv2.setMouseCallback("Original Image ", mousePoints)
    #     # Refreshing window all time
    #     cv2.waitKey(1)
    #     if endloop:
    #         break
    #     if counter>=4:
    #         break

    # cv2.destroyAllWindows()
    # print(point_matrix)
    # bottom_left = [int(bottomLeft_x), int(bottomLeft_y)]
    # top_left = [int(topLeft_x), int(topLeft_y)]
    # bottom_right = [int(bottomRight_x), int(bottomRight_y)]
    # top_right = [int(topRight_x), int(topRight_y)]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)


    copied = np.copy(blurred)
    cv2.line(copied, tuple(bottom_left), tuple(bottom_right), (255, 0, 0), 5)
    cv2.line(copied, tuple(bottom_right), tuple(top_right), (255, 0, 0), 5)
    cv2.line(copied, tuple(top_left), tuple(bottom_left), (255, 0, 0), 5)
    cv2.line(copied, tuple(top_left), tuple(top_right), (255, 0, 0), 5)

    copied = np.copy(blurred)


    src = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    interested = cv2.warpPerspective(copied, M, (width, height)) # Image warping
    stframe.image(interested, use_column_width=True)
    # cv2.imshow("ROI",interested)
    # cv2.waitKey(0)

    

    # interested = region_of_interest(copied, vertices)
    # final_mask = region_of_interest(image, vertices)
    wheres = np.where(interested > 80)
    # clustered = AgglomerativeClustering(4).fit_predict(wheres[1].reshape([-1, 1]))
    clustered = KMeans(n_clusters =4, random_state=0).fit_predict(wheres[1].reshape([-1, 1]))


    # actually you can calculate the angle, from there we can determine the position of min and max

    yellow_x, yellow_y = wheres[1][clustered == 3], wheres[0][clustered == 3]
    yellow_top = [np.max(yellow_x), np.min(yellow_y)]
    yellow_bottom = [np.min(yellow_x), np.max(yellow_y)]

    blue_x, blue_y = wheres[1][clustered == 2], wheres[0][clustered == 2]
    blue_top = [np.max(blue_x), np.min(blue_y)]
    blue_bottom = [np.min(blue_x), np.max(blue_y)]

    green_x, green_y = wheres[1][clustered == 1], wheres[0][clustered == 1]
    green_top = [np.max(green_x), np.min(green_y)]
    green_bottom = [np.min(green_x), np.max(green_y)]

    red_x, red_y = wheres[1][clustered == 0], wheres[0][clustered == 0]
    red_top = [np.max(red_x), np.min(red_y)]
    red_bottom = [np.min(red_x), np.max(red_y)]

    min_y_point = min(np.min(blue_y),np.min(green_y),np.min(red_y),np.min(yellow_y))
    max_y_point = max(np.max(blue_y), np.max(green_y), np.max(red_y), np.max(yellow_y))

    copied = np.copy(interested)
    m_blue, c_blue = np.polyfit([blue_top[1], blue_bottom[1]],
                                [blue_top[0], blue_bottom[0]], 1)
    dot_blue_bottom = [int(image.shape[0] * m_blue + c_blue), image.shape[0]]
    dot_blue_top = [int(green_top[1] * m_blue + c_blue), green_top[1]]
    cv2.line(copied, tuple(dot_blue_top), tuple(dot_blue_bottom), (255, 0, 0), 10)


    m_green, c_green = np.polyfit([green_top[1], green_bottom[1]],
                                  [green_top[0], green_bottom[0]], 1)
    dot_green_bottom = [int(image.shape[0] * m_green + c_green), image.shape[0]]
    cv2.line(copied, tuple(green_top), tuple(dot_green_bottom), (255, 0, 0), 10)

    m_red, c_red = np.polyfit([red_top[1], red_bottom[1]],
                              [red_top[0], red_bottom[0]], 1)
    dot_red_bottom = [int(image.shape[0] * m_red + c_red), image.shape[0]]
    cv2.line(copied, tuple(red_top), tuple(dot_red_bottom), (255, 0, 0), 10)

    cv2.line(copied, (np.max(yellow_x), min_y_point), (np.min(yellow_x),max_y_point), (255, 0, 0), 4)
    cv2.line(copied, (np.max(blue_x), min_y_point), (np.min(blue_x),max_y_point), (255, 0, 0), 4)
    cv2.line(copied, (np.max(green_x), min_y_point), (np.min(green_x),max_y_point), (255, 0, 0), 4)
    cv2.line(copied, (np.max(red_x), min_y_point), (np.min(red_x),max_y_point), (255, 0, 0), 4)

    cv2.line(copied,(0, min_y_point+10),(width, min_y_point+10),(255,250,250),5)
    cv2.line(copied,(0,max_y_point-10),(width,max_y_point-10),(255,250,250),5)


    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst = np.float32([top_left, top_right, bottom_right, bottom_left])
    Minv = cv2.getPerspectiveTransform(src, dst) # Inverse transformation
    copied = cv2.warpPerspective(copied, Minv, (width, height))
    stframe.image(copied, use_column_width=True)
    # cv2.imshow("results",copied)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(copied, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # copied, contours = markSections.make_rectangles(backtorgb)
    print("Number of Contours found = " + str(len(contours)))
    # Draw all contours
    # -1 signifies drawing all contours
    backtorgb = cv2.cvtColor(copied,cv2.COLOR_GRAY2RGB)
    cnts= []
    for c in contours:
        # if cv2.contourArea(c)>10000:
            cv2.drawContours(backtorgb, c, -1, (0, 255, 0), 3)
            cnts.append(c)
    stframe.image(backtorgb, use_column_width=True)
    # cv2.imshow("results",backtorgb)
    # cv2.waitKey(0)


    # print(image.shape, backtorgb.shape)
    # final = cv2.addWeighted(image, 1, backtorgb, 1, 0)
    # scanLanes.is_car_there(final,cnts)

    # # copied = cv2.merge((image, copied))
    # cv2.imshow("Final",final)
    # cv2.waitKey(0)



    cap = cv2.VideoCapture(file)
    while True:
        ret, original_frame = cap.read()
        if ret is False:
            break
        original_frame = cv2.resize(original_frame,(1080, 720))
        lane_image = np.copy(original_frame)
        # cv2.imshow("L",lane_image)
        # cv2.imshow("B",backtorgb)
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break
        
        # final = cv2.addWeighted(lane_image, 1, backtorgb, 1, 0)
        scanLanes.is_car_there(original_frame,cnts, cropimage, plateimage,image_files)
        # cv2.imshow("Final",final)
        final = cv2.addWeighted(original_frame, 1, backtorgb, 1, 0)
        stframe.image(final, channels="BGR", use_column_width=True)
        # cv2.imwrite(r"C:\Users\rohan\Downloads\oVERT\Github\Home-GUI\static\img\output.jpg",final)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()