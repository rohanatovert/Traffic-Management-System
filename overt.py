

from Helmet_Sai.phu_video import main as helmetMain
from RLVD_pavan.rlvd_track_pavan import main as rlvdMain
from LaneViolation_rohan.finalPro import main as laneMain
from LaneViolation_rohan.finalPro import get_road_map
from SpeedTracking_rohan.track import main as speedMain
from Vehicle_Counting.track import detect as countMain
import tempfile
import cv2
import torch
import streamlit as st
import os

from PIL import Image
import argparse
from pathlib import Path
import streamlit.components.v1 as components
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
OPERATION = ""
frame = None

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='videos/motor.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

if __name__ == '__main__':
    im = Image.open(f"Home_GUI/assets/overt logo.png")
    st.set_page_config(
    page_title="Traffic Management App",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded")

    st.title('Traffic Rules Violation Detection System')
    st.markdown('<h3 style="color: red"> OVERT IDEAS AND SOLUTIONS </h3', unsafe_allow_html=True)
    html = f'''
    <style>
        #scrollable {{
        
            height: 300px;
    
            background-color: #000000;
            overflow-y: scroll;
        }}
        
    </style>
    '''
    # Display the HTML page in Streamlit
    st.write(html, unsafe_allow_html=True)

    
    st.markdown('**VIDEO**')

    # upload video
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])

    if video_file_buffer:
        st.sidebar.text('Input video')
        st.sidebar.video(video_file_buffer)
        # save video from streamlit into "videos" folder for future detect
        with open(os.path.join('videos', video_file_buffer.name), 'wb') as f:
            f.write(video_file_buffer.getbuffer())

    option = st.sidebar.selectbox(
        'Select the option',
        ('RLVD', 'Helmet Violation', 'Lane Violation', 'Speed Violation', 'Count Vehicles'))
    st.sidebar.write('You selected:', option)

    
    stream, output, plates = st.columns([2,1,1])
    
    with stream:
        status = st.empty()
        stframe = st.empty()
        
        if video_file_buffer is None:
            status.markdown('<font size= "4"> **Status:** Waiting for input </font>', unsafe_allow_html=True)
        else:
            status.markdown('<font size= "4"> **Status:** Ready </font>', unsafe_allow_html=True)
  

    with output:
        st.markdown('**VIOLATIONS**')
        cropimage = st.empty()
        plateimage = st.empty()
        # plateimage = st.empty()
        # Create a container with a vertical scrollbar
        # container = st.container()
        # container.vertical_scrollbar = True
    with plates:
        st.markdown('**Plates**')

        # directory_path = r'Home_GUI\static\img\plates'

        # # Use the os module to list all files in the directory
        # files = os.listdir(directory_path)

        # # Filter out directories and other non-file items
        # image_files = [fr"Home_GUI\static\img\plates\{f}" for f in files if os.path.isfile(os.path.join(directory_path, f))]
        

        # Generate the HTML code for the images
        # image_html = ""
        image_files = st.empty()
        # print(image_files)
        # for file in image_files:
        #     image_html += st.image(file)
            # if file.endswith(".jpg") or file.endswith(".png"):
                
            #     # image_path = os.path.join(IMAGE_FOLDER, file)
                
            #     file_ = open(file, 'rb')
            #     contents = file_.read()
            #     data_url = base64.b64encode(contents).decode("utf-8")
            #     image_html += f'<img src="data:image/gif;base64,{data_url}" alt="{file}">'
                
        # Generate the HTML page with the images and scrollbar
        
        # html = f'''
        #     <div id="scrollable">
        #         {image_files}
        #         plateimage = st.empty()
        #     </div>
        # '''

        # # Display the HTML page in Streamlit
        # st.markdown(html, unsafe_allow_html=True)   
        # # st.sidebar.write(st.session_state)
        

    track_button = st.sidebar.button('START')
    # reset_button = st.button('RESET ID')
    if option == "Lane Violation":
        if video_file_buffer is not None:
            
            def draw_lines(image,  xtop, ytop, topWidth, xbottom,  ybottom, bottomWidth):
                height, width, _ = image.shape
                x1, y1 = xtop, ytop
                x2, y2 = xbottom,  ybottom
                cv2.line(image, (x1, y1), (topWidth, y1), (0, 255, 0), 5)
                cv2.line(image, (x2, y2), (bottomWidth, y2), (0, 255, 0), 5)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cv2.line(image, (topWidth, y1), (bottomWidth, y2), (0, 255, 0), 5)
                return image

            try:
                name = os.path.basename(video_file_buffer.name).split(".")[0] + '.jpg'
                # print("Looking for ",name)
                image = cv2.imread(name)
                image = cv2.resize(image,(1080, 720))
                # print("Got it!")
            except:
                image= get_road_map(f'videos/{video_file_buffer.name}', stframe)
            xtop = st.sidebar.slider("X Top", 0, image.shape[0], image.shape[0]//2, 1)
            ytop = st.sidebar.slider("Y Top", 0, image.shape[0], image.shape[0]//2, 1)
            topWidth = st.sidebar.slider("Top Width", 0, image.shape[0]*2, image.shape[0]//2+150, 1)

            xbottom = st.sidebar.slider("X Bottom", 0, image.shape[0], image.shape[0]//2-100, 1)
            ybottom = st.sidebar.slider("Y Bottom", 0, image.shape[0], image.shape[0]//2+200, 1)
            bottomWidth = st.sidebar.slider("Bottom Width", 0, image.shape[0]*2, image.shape[0]//2+200, 1)

            line_image = draw_lines(image.copy(), xtop, ytop, topWidth, xbottom,  ybottom, bottomWidth)
            stframe.image(line_image, channels="BGR")




    st.sidebar.markdown('---')
    st.sidebar.title('Settings')

    # setting hyperparameter
        
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)
    line = st.sidebar.number_input('Line position', min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    st.sidebar.markdown('---')

    custom_class = st.sidebar.checkbox('Custom classes')
    assigned_class_id = []
    names = ['car', 'motorcycle','bus', 'truck']

    # custom classes
    if custom_class:
        assigned_class = st.sidebar.multiselect('Select custom classes', list(names))
        for each in assigned_class:
            assigned_class_id.append(names.index(each))
    


    
    if track_button:
        conf_thres = confidence
        filename = f'videos/{video_file_buffer.name}'
        parser = argparse.ArgumentParser()
        
        opt = parse_opt()
        # parser.add_argument('--source', nargs='+', type=str, default=r'C:\Users\rohan\Downloads\oVERT\Github\Helmet_Sai\biker_yolov5s.pt', help='model.pt path(s)')
        # parser.add_argument('--conf_thres', nargs='+', type=str, default=confidence, help='model.pt path(s)')
        # opt.conf_thres = confidence
        opt.source = f'videos/{video_file_buffer.name}'
        status.markdown('<font size= "4"> **Status:** Running... </font>', unsafe_allow_html=True)
        if option== "RLVD":
            print("Doing RLVD...")
            opt.yolo_weights=r'yolov5\yolov5s.pt'
            opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
            opt.tracking_method='bytetrack'
            opt.tracking_config = r"RLVD_pavan\trackers\bytetrack\configs\bytetrack.yaml"
            opt.save_crop=True
            opt.show_vid=True
            rlvdMain( opt , stframe, cropimage, plateimage, image_files)
            # os.system(rf'python C:\Users\rohan\Downloads\oVERT\Github\RLVD_pavan\rlvd_track_pavan.py --source {filename} --show-vid --yolo-weights C:\Users\rohan\Downloads\oVERT\Github\Helmet-sai\yolov5s.pt --save-crop --save-vid')
            # rlvd.run(filename)
        if option =="Helmet Violation":
            print("Doing Helmet Violation...")
            # parser.add_argument('--catching', action='store_true', default = True, help='existing project/name ok, do not increment')
            # parser.add_argument('--model', nargs='+', type=str, default=r'C:\Users\rohan\Downloads\oVERT\Github\Helmet_Sai\biker_yolov5s.pt', help='model.pt path(s)')
            opt.model = r'Helmet_Sai\biker_yolov5s.pt'
            opt.catching = True
            helmetMain( opt , stframe, cropimage, plateimage, image_files)
        if option =="Lane Violation":
            print("Doing Lane Detection...")
            laneMain( opt , image, (xbottom, ybottom), (xtop, ytop), (topWidth, ytop), (bottomWidth, ybottom), stframe, cropimage, plateimage, image_files)
            # os.system(rf'python LaneViolation-rohan\finalPro.py -i {filename}')
        if option =="Speed Violation":
            print("Doing Speed Violation...")
            
            # parser.add_argument('--yolo-weights', nargs='+', type=str, default=r'C:\Users\rohan\Downloads\oVERT\Github\yolov5\yolov5s.pt', help='model.pt path(s)')
            # parser.add_argument('--classes', nargs='+', type=int, default = 2, help='filter by class: --classes 0, or --classes 0 2 3')
            opt.yolo_weights = Path(r'yolov5\yolov5s.pt')
            opt.classes = 2
            speedMain( opt , stframe, cropimage, plateimage, image_files)

            # os.system(rf'python C:\Users\rohan\Downloads\oVERT\Github\SpeedTracking-rohan\track.py --source {filename} --yolo-weights C:\Users\rohan\Downloads\oVERT\Github\SpeedTracking-rohan\yolov5s.pt --classes 2 --save-crop --show-vid')
        if option =="Count Vehicles":
            print("Doing COUNTING...")
            opt.yolo_model=r'Vehicle_Counting\best64.pt'
            opt.deep_sort_model='osnet_x0_25'
            opt.fourcc = 'mp4v'
            opt.evaluate=True
            opt.config_deepsort=r"Vehicle_Counting\deep_sort\configs\deep_sort.yaml"
            opt.output='inference/output'  # output folder
            # parser.add_argument('--source', type=str, default='videos/motor.mp4', help='source')  # file/folder, 0 for webcam
            # opt = parser.parse_args()
            car, bus, truck, motor = st.columns(4)
            with car:
                st.markdown('**Car**')
                car_text = st.markdown('__')
            
            with bus:
                st.markdown('**Bus**')
                bus_text = st.markdown('__')

            with truck:
                st.markdown('**Truck**')
                truck_text = st.markdown('__')
            
            with motor:
                st.markdown('**Motorcycle**')
                motor_text = st.markdown('__')
            
            fps, _,  _, _  = st.columns(4)
            with fps:
                st.markdown('**FPS**')
                fps_text = st.markdown('__')

            # detect(opt, stframe, cropimage, car_text, bus_text, truck_text, motor_text, line, fps_text)
            countMain( opt, stframe, cropimage, car_text, bus_text, truck_text, motor_text, line, fps_text)
            # os.system(rf'python Helmet-sai\detect.py --weights C:\Users\rohan\Downloads\oVERT\Github\ANPR-rohan\new_weights.py\best.pt --img 416 --conf 0.4 --source {filename} --save-crop --view-img')
        
        status.markdown('<font size= "4"> **Status:** Finished ! </font>', unsafe_allow_html=True)
        end_noti = st.markdown('<center style="color: blue"> FINISH </center>',  unsafe_allow_html=True)

        
        # reset ID and count from 0
        # reset()
        
        
        # with torch.no_grad():
        #     detect(opt, stframe, car_text, bus_text, truck_text, motor_text, line, fps_text)
    
    # if reset_button:
    #     # reset()
    #     st.markdown('<h3 style="color: blue"> Reseted ID </h3>', unsafe_allow_html=True)
    # javascript_code = """
    #         console.log("HELOOW");
    #         var image_div = document.getElementById("scrollMe");
    #         var no_of_images_already = image_div.children.length;
    #         var list_of_src_already = [];
    #         var images_already = image_div.children;
    #         for(var i=0; i< no_of_images_already; i++) {  
    #             image_name = images_already[i].src.replace(/^.*[\\\/]/, '');
    #             list_of_src_already.push(image_name)
    #             }
    #         var new_list_of_images = [];
    #         for(var i=0; i< list_of_images.length; i++) {
    #             image_name = list_of_images[i].replace(/^.*[\\\/]/, '');
    #             // console.log(image_name,list_of_src_already, list_of_src_already.includes(image_name));
    #             if (!(list_of_src_already.includes(image_name))){
    #             new_list_of_images.push(list_of_images[i])
    #             var img = new Image();
    #             img.src = list_of_images.at(-1);
    #             image_div.insertBefore(img,image_div.firstChild);
    #         }
    #         }
    #     """
    # # Add the JavaScript code to the Streamlit app
    # st.write('<script>' + javascript_code + '</script>', unsafe_allow_html=True)
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # Define the directory path
