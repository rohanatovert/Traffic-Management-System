import argparse
from PIL import Image
import os
import time
import sys
import platform
import numpy as np
from pathlib import Path
import torch
import base64
import torch.backends.cudnn as cudnn
# m = torch.hub.load(r'C:\Users\rohan\Downloads\oVERT\Github\Helmet-sai\Yolov5_DeepSort_Pytorch\yolov5', 'custom', path=r'C:\Users\rohan\Downloads\oVERT\Github\Helmet-sai\yolov5s.pt', source='local')
# m.classes = [2,3,7,9]
import sys
sys.path.insert(0,r'RLVD_pavan')
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
crop_plate_dir = Path(r"Github\Home-GUI\static\img\plates")
model_plates = torch.hub.load(r'yolov5', 'custom', path=r'detection_best.pt', source='local') # ADDED

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.segment.general import masks2segments, process_mask, process_mask_native
from trackers.multi_tracker_zoo import create_tracker
import trafficLightColor
from easyocr import Reader
from torchvision.transforms import Resize as rs

def getLightThresh(im, xlight, ylight, wlight, hlight, showStages=False):
    frame = im
    if(showStages==True):
        cv2.imshow('fr',frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
    temp =frame.copy()
    temp2=frame.copy()
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(grayscaled, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.erode(th, kernel, iterations=1)
    th = cv2.dilate(th, kernel, iterations=2)
    if(showStages==True):
        cv2.imshow('thresh',th)
        cv2.waitKey()
        cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # cv2.imshow('cont',ct)
    # cv2.waitKey()
    cv2.destroyAllWindows()
    # im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contIndex = 0
    allContours = []
    for contour in contours:
        M = cv2.moments(contour)
        if(cv2.contourArea(contour)>800):
            # print('length: ',len(contour))
            if(len(contour)<100):
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                if(len(approx)==4):
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.drawContours(frame, contours, contIndex, (0, 255, 0), 3)
                    cv2.rectangle(temp,(x,y),(x+w,y+h),(255,0,0),2)
                    allContours.append((x,y,w,h))
        contIndex = contIndex + 1

    cv2.drawContours(temp2, contours, -1, (0, 255, 0), 3)
    if(showStages==True):
        cv2.imshow('frame',temp2)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.imshow('frame',frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.imshow('frame',temp)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # cv2.imshow('x',frame)
    # cv2.waitKey()
    frame=temp
    minIndex = 0
    count=0
    minDistance = 10000000
    for rect in allContours:
        x,y,w,h = rect
        if(ylight+wlight<y):
        #  print(x)
        #  print(y)
        #  print(xlight)
        #  print(ylight)
         cv2.line(temp, (xlight,ylight), (x, y), (0, 0, 255), 2)
         if (((x-xlight)**2 + (y-ylight)**2)**0.5) < minDistance:
             minDistance = (((x-xlight)**2 + (y-ylight)**2)**0.5)
             minIndex=count
        count=count+1
    (x,y,w,h) = allContours[minIndex]
    if(showStages==True):
        cv2.imshow('with-dist',temp)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.line(temp, (0, y), (1300, y), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.imshow('with-dist',temp)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return y




reader = Reader(['en'])

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        stframe=None,
        cropimage=None,
        plateimage=None,
        image_files = None
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    yolo_weights = Path(yolo_weights)
    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources
    
    

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    #model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    ylight = None
    redLightViolatedCounter = {}
    tracking = {}
    image_html = ""
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        
        with dt[0]:
            im = torch.from_numpy(im).to(device) # line 119
            r = rs((736, 1280))
            im = r(im)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            if is_seg:
                pred, proto = model(im, augment=augment, visualize=visualize)[:2]
            else:
                pred = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        # classes = 2,3,7,9 #ADDED
        with dt[2]:
            if is_seg:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
            else:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            im0 = cv2.resize(im0.copy(), (1280,736))
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                if is_seg:
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                        masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                    else:
                        masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                ##########################################################3
                if ylight==None: 
                    for *xyxy, conf, cls in reversed(det):
                        # print(names[int(cls)])
                        if str(names[int(cls)]).lower()=="traffic light":
                            print("Found a traffic light!")
                            if conf.cpu()>=0.5:
                                xmin, ymin, xmax, ymax = xyxy
                                xmin = int(xmin)
                                ymin = int(ymin)
                                xmax = int(xmax)
                                ymax = int(ymax) 
                                width = xmax-xmin
                                height = ymax-ymin

                                cv2.rectangle(im0,(xmin,ymin),(xmax,ymax),(0,0,255),2)
                                # cv2.imshow("trafficlights",im0)
                                # cv2.waitKey(0)
                                xlight,ylight,wlight,hlight = xmin,ymin,width,height
                            
                                thresholdRedLight = getLightThresh(im0, xlight,ylight,wlight,hlight)
                                # thresholdRedLight += 10
                                break
                ######################################################################################
                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output) in enumerate(outputs[i]):
    
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        if save_txt:
                            # to MOT format
                            
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            if str(names[int(c)]).lower()!="traffic light":
                                annotator.box_label(bbox, label, color=color)
                            if is_seg:
                                # Mask plotting
                                annotator.masks(
                                    masks,
                                    colors=[colors(x, True) for x in det[:, 5]],
                                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                                    255 if retina_masks else im[i]
                                )
                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            
                        
                        ###########################################################################
                        if(ylight!=None):
                            light = imc[ylight:ylight + hlight, xlight:xlight + wlight]
                            # cv2.imshow("lights",light)
                            # b, g, r = cv2.split(light)
                            # light=cv2.merge([r,g,b])
                            
                            if(trafficLightColor.estimate_label(light)=="green"):
                                # print('GREEN LIGHT')
                                pass
                            else:
                                ## ADDED
                                cv2.line(im0, (0, thresholdRedLight), (1300, thresholdRedLight), (0, 0, 0), 4, cv2.LINE_AA)
                                cv2.putText( im0, 'Violation Counter: ' + str(len(redLightViolatedCounter)), (30, 60),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4, cv2.LINE_AA)
                                cv2.putText( im0, trafficLightColor.estimate_label(light), (xlight, ylight-20),
                                                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                                
                                
                                if names[int(c)].lower() in ["car","truck","motorcycle"]:
                                    xmin, ymin, xmax, ymax = output[0], output[1], output[2],output[3]
                                    # print(xmin, ymin, xmax, ymax)
                                    xmid = int((xmax+xmin)/2)
                                    ymid = int((ymax+ymin)/2)
                                    xmin = int(xmin)
                                    ymin = int(ymin)
                                    xmax = int(xmax)
                                    ymax = int(ymax) 
                                    
                                    # print(names[int(c)].lower(), id)
                                    if id not in tracking:
                                        tracking[id] = [xmid,ymid]
                                    else:
                                        new_coords= xmid,ymid
                                        is_going_up = tracking[id][1]>new_coords[1] and new_coords[1]>thresholdRedLight
                                        tracking[id] = [xmid,ymid]
                                        # cv2.putText(im0, f"{id} {new_coords[1]}", (xmid,ymid),
                                        #                             cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                                        if is_going_up:
                                            if ymin > thresholdRedLight:
                                                cv2.rectangle( im0, (xmax, ymax), (xmin, ymin), (0, 255, 0), 3)
                                            if(ymin < thresholdRedLight):
                                                if save_crop:
                                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                                    save_one_box(bbox, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                                                    vehicle_crop = imc[ymin:ymax, xmin:xmax]
                                                    # cv2.imshow("Violation",vehicle_crop)
                                                    
                                                    cropimage.image(vehicle_crop, channels="BGR")
                                                    results = model_plates(vehicle_crop) #ADDED
                                                    if 0 in results.pandas().xyxy[0]['class'] and results.pandas().xyxy[0]['confidence'].values[0]>=0.3:
                                                        # if id not in tracking:
                                                            crop = results.crop(save=False)[0]["im"]
                                                            # cv2.imshow('crop', crop)
                                                            # cv2.imwrite(rf"Home-GUI\static\img\plates\plate{int(time.time())}.jpg", crop)
                                                            plateimage.image(crop, channels="BGR",  width = 400)
                                                            ret, jpeg = cv2.imencode('.jpg', crop)
                                                            jpg_as_text = base64.b64encode(jpeg).decode('utf-8')
                                                            
                                                            image_html += f'<img src="data:image/gif;base64,{jpg_as_text}" alt="plate">'
                                                            html = f'''
                                                                <div id="scrollable">
                                                                    {image_html}
                                                                </div>
                                                            '''
                                                            image_files.empty()
                                                            image_files.write(html,unsafe_allow_html=True)
                                                            
                                                            # save_one_box(crop, vehicle_crop, file=crop_plate_dir / 'plate.jpg', BGR=True)
                                                            detection = reader.readtext(crop)
                                                            if len(detection)!=0 and detection[0][2]>=0.3:
                                                                text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
                                                                print(text)
                                                cv2.putText( im0, 'Violation', (xmin,ymax), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
                                                cv2.rectangle( im0, (xmax, ymax), (xmin, ymin), (0, 0, 255), 3) 
                                                redLightViolatedCounter[id] = tracking[id]
                                    
                                                    # if 0 in results.pandas().xyxy[0]['class'] and results.pandas().xyxy[0]['confidence'].values[0]>=0.3:
                                                    #     crop = results.crop(save=False)[0]["im"]
                                                    #     cv2.imshow('crop', crop)
                                                    #     detection = reader.readtext(crop)
                                                    #     if len(detection)!=0 and detection[0][2]>=0.3:
                                                    #         text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
                                                    #         print(text)
                                                    #         # results.save() #ADDED
                                                    #         # results.show()'''
                                    #############################################       
            else:
                    pass
                    #tracker_list[i].tracker.pred_n_update_all_tracks()
                    
            # Stream results
            im0 = annotator.result()
            
            


            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                
                # cv2.imshow(str(p), im0)
                # cv2.imwrite(r"C:\Users\rohan\Downloads\oVERT\Github\Home-GUI\static\img\output.jpg",im0)
                stframe.image(im0, channels="BGR", use_column_width=True)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]                
                
                
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")
        
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
    # os.system(rf"python Helmet-sai\detect_number_plate.py --weights Helmet-sai\best_number_plate.pt --source {save_dir}/crops/violation --img 640 --conf 0.4 --view-img --save-crop")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5s-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt, stframe, cropimage, plateimage, image_files):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt), stframe = stframe, cropimage = cropimage, plateimage = plateimage, image_files=  image_files)


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
