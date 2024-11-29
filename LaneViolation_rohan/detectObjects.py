import cv2
from easyocr import Reader
import torch
model_plates = torch.hub.load(r'yolov5', 'custom', path=r'yolov5\best_number_plate.pt', source='local') # ADDED
# model2 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# # model2.conf = 0.5

reader = Reader(['en'])
import base64
center_points = []
image_html = ""

def detect(frame,BIKE_LANE,lane, cropimage, plateimage, image_files):
    
    height, width = frame.shape[0], frame.shape[1]
   
    # cropped = frame[300:height, 400: 800] #manually
    
    car_cascade = cv2.CascadeClassifier(r'LaneViolation_rohan\Cascades\Vehicle and pedestrain detection\cars.xml')
    bike_cascade = cv2.CascadeClassifier(r'LaneViolation_rohan\Cascades\Vehicle and pedestrain detection\two_wheeler.xml')

    # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur=cv2.blur(frame,(3,3))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 9)
    bikes = bike_cascade.detectMultiScale(gray, 1.1, 9)
    # results = model2(frame)
    # detections = results.pandas().xyxy[0]
    # car_bboxes = detections[detections['name'] == 'car'][['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    # bike_bboxes = detections[detections['name'] == 'motorbike'][['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    # for bbox in car_bboxes:
    #     cv2.imshow("Car", frame[int(bbox[1]):int(bbox[3]-bbox[1]),int(bbox[0]):int(bbox[2]-bbox[0])])
    #     cv2.waitKey(1)
    # for bbox in bike_bboxes:
    #     cv2.imshow("Bike", frame[int(bbox[1]):int(bbox[3]-bbox[1]),int(bbox[0]):int(bbox[2]-bbox[0])])
    #     cv2.waitKey(1)
    for (x,y,w,h) in cars:
        # plate = frame[y+300:y + 300 + h, x+ 400:x +400 + w] #manual
        # cv2.rectangle(frame,(x+ 400,y+300),(x+ 400 +w, y+300 +h) ,(51 ,51,255),2) #manual
        # cv2.rectangle(frame, (x+ 400, y+300 - 40), (x + 400+ w, y+300), (51,51,255), -2) #manual
        # cv2.putText(frame, 'Car', (x+ 400, y+300 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) #manual
        # cv2.imshow('car',plate)
        vehicle_crop = frame[y:y + h, x:x + w] #manual
        
        if BIKE_LANE:
            cv2.rectangle(frame,(x,y),(x +w, y+h) ,(51 ,51,255),2) #manual
            cv2.rectangle(frame, (x, y- 40), (x+ w, y), (51,51,255), -2) #manual
            cv2.putText(frame, f'VIOLATION:{lane}', (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) #manual
            # cv2.imshow('car',plate)
            cropimage.image(vehicle_crop, channels="BGR",  width = 400)
            # cv2.imwrite(r"C:\Users\rohan\Downloads\oVERT\Github\Home-GUI\static\img\crops\crop.jpg", plate)
            results = model_plates(vehicle_crop) #ADDED
            if 0 in results.pandas().xyxy[0]['class'] and results.pandas().xyxy[0]['confidence'].values[0]>=0.3:
                plate = results.crop(save=False)[0]["im"]
                # cv2.imshow('crop', crop)
                # cv2.imwrite(rf"C:\Users\rohan\Downloads\oVERT\Github\Home-GUI\static\img\plates\plate{int(time.time)}.jpg", crop)
                plateimage.image(plate, channels="BGR",  width = 400)
                ret, jpeg = cv2.imencode('.jpg', plate)
                jpg_as_text = base64.b64encode(jpeg).decode('utf-8')
                
                image_html += f'<img src="data:image/gif;base64,{jpg_as_text}" alt="plate">'
                html = f'''
                    <div id="scrollable">
                        {image_html}
                    </div>
                '''
                image_files.empty()
                image_files.write(html,unsafe_allow_html=True)
                detection = reader.readtext(plate)
                if len(detection)!=0 and detection[0][2]>=0.3:
                    text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
                    print(text)
        else:
            cv2.rectangle(frame,(x,y),(x +w, y+h) ,(51 ,255,51),2) #manual
            cv2.rectangle(frame, (x, y- 40), (x+ w, y), (51,255,51), -2) #manual
            cv2.putText(frame, f'Car:{lane}', (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) #manual
        
        # cx = int((x + x + w)/2)
        # cy = int((y + y + h)/2)
        # # center_points.append((cx,cy))
        # cv2.circle(frame,(cx,cy),5,(0,0,255), -1)
       


    for (x,y,w,h) in bikes:
        vehicle_crop = frame[y:y + h, x:x + w] #manual
        
        if not BIKE_LANE:
            cv2.rectangle(frame,(x,y),(x +w, y+h) ,(51 ,51,255),2) #manual
            cv2.rectangle(frame, (x, y- 40), (x+ w, y), (51,51,255), -2) #manual
            cv2.putText(frame, 'VIOLATION', (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) #manual
            # cv2.imshow('bike',plate)
            cropimage.image(vehicle_crop, channels="BGR",  width = 400)
            # cv2.imwrite(r"C:\Users\rohan\Downloads\oVERT\Github\Home-GUI\static\img\crops\crop.jpg", plate)
            results = model_plates(vehicle_crop) #ADDED
            if 0 in results.pandas().xyxy[0]['class'] and results.pandas().xyxy[0]['confidence'].values[0]>=0.3:
                plate = results.crop(save=False)[0]["im"]
                # cv2.imshow('crop', crop)
                # cv2.imwrite(rf"C:\Users\rohan\Downloads\oVERT\Github\Home-GUI\static\img\plates\plate{int(time.time)}.jpg", crop)
                ret, jpeg = cv2.imencode('.jpg', plate)
                jpg_as_text = base64.b64encode(jpeg).decode('utf-8')
                
                image_html += f'<img src="data:image/gif;base64,{jpg_as_text}" alt="plate">'
                html = f'''
                    <div id="scrollable">
                        {image_html}
                    </div>
                '''
                image_files.empty()
                image_files.write(html,unsafe_allow_html=True)
                detection = reader.readtext(plate)
                if len(detection)!=0 and detection[0][2]>=0.3:
                    text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
                    print(text)
        else:
            cv2.rectangle(frame,(x,y),(x +w, y+h) ,(51 ,255,51),2) #manual
            cv2.rectangle(frame, (x, y- 40), (x+ w, y), (51,255,51), -2) #manual
            cv2.putText(frame, 'Bike', (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) #manual
      
      
        
