from tkinter import *
from ttkbootstrap.constants import *
import ttkbootstrap as tb
from tkinter import filedialog
import cv2 
from PIL import Image, ImageTk
import os

root = tb.Window(themename="superhero")
root.title("OVERT Traffic Management")
root.geometry("1000x800")

#FUNCTIONS
counter = 0
def changer():
    global counter
    counter +=1
    if counter%2==0:
        my_button.config(text="Pause")
    else:
        my_button.config(text="Resume")
    if var1.get()==1: #RLVD
        print("Doing RLVD...")
        # os.system(rf'python RLVD-pawan\yolo_video_new.py --input {input_path} --output output/airport_output2.avi')
        os.system(rf'python c:/Users/rohan/Downloads/oVERT/Github/RLVD_pavan/rlvd_track_pavan.py --source {input_path} --show-vid --yolo-weights C:\Users\rohan\Downloads\oVERT\Github\Helmet-sai\yolov5s.pt --save-crop --save-vid')
    if var2.get()==1: #Overspeeding
        print("Doing Speed Violation...")
        # os.system(rf'python SpeedDetection-rohan\main.py --input {input_path}')
        os.system(rf'python C:\Users\rohan\Downloads\oVERT\Github\SpeedTracking-rohan\track.py --source {input_path} --yolo-weights C:\Users\rohan\Downloads\oVERT\Github\SpeedTracking-rohan\yolov5s.pt --classes 2 --show-vid')
    if var3.get()==1: #Lane Violation
        print("Doing Lane Detection...")
        os.system(rf'python LaneViolation-rohan\finalPro.py -i {input_path}')
    if var4.get()==1: #Helmet Violation
        print("Doing Helmet Violation...")
        # os.system(rf'python Helmet-sai\detect.py --weights Helmet-sai\best_helmet_detection.pt --source {input_path} --img 640 --conf 0.4 --view-img --save-crop')
        # os.system(rf'python C:\Users\rohan\Downloads\oVERT\Github\Helmet-sai\Yolov5_DeepSort_Pytorch\track.py --source {input_path} --yolo-weights "C:\Users\rohan\Downloads\oVERT\Github\Helmet-sai\best_helmet_detection.pt" --img 640  --show-vid --save-crop --save-vid --conf-thres 0.80')
        os.system(rf'python C:\Users\rohan\Downloads\oVERT\Github\Helmet-sai\Yolov5_DeepSort_Pytorch\track.py --source {input_path} --yolo-weights "C:\Users\rohan\Downloads\oVERT\Github\Helmet-sai\biker_yolov5m.pt" --img 640  --show-vid --save-crop --save-vid --conf-thres 0.80')
    if var5.get()==1: #ANPR
        print("Doing ANPR...")
        os.system(rf'python Helmet-sai\detect.py --weights C:\Users\rohan\Downloads\oVERT\Github\ANPR-rohan\new_weights.py\best.pt --img 416 --conf 0.4 --source {input_path} --save-crop --view-img')


def checker1():
    global var1, var2, var3, var4, var5
    var2.set(0)
    var3.set(0)
    var4.set(0)
    var5.set(0)
    if var1.get() == 1:    
        my_check1.config(bootstyle="danger, toolbutton")
        print(var1.get(), var2.get(), var3.get(), var4.get(), var5.get())
    else:
        my_check1.config(bootstyle="primary, toolbutton")

def checker2():
    global var1, var2, var3, var4, var5
    var1.set(0)
    var3.set(0)
    var4.set(0)
    var5.set(0)
    if var2.get() ==1:
        my_check2.config(bootstyle="danger, toolbutton")
        print(var1.get(), var2.get(), var3.get(), var4.get(), var5.get())
    else:
        my_check2.config(bootstyle="primary, toolbutton")
        
def checker3():
    global var1, var2, var3, var4, var5
    var1.set(0)
    var2.set(0)
    var4.set(0)
    var5.set(0)
    if var3.get() ==1:
        my_check3.config(bootstyle="danger, toolbutton")
        print(var1.get(), var2.get(), var3.get(), var4.get(), var5.get())
    else:
        my_check3.config(bootstyle="primary, toolbutton")

def checker4():
    global var1, var2, var3, var4, var5
    var1.set(0)
    var2.set(0)
    var3.set(0)
    var5.set(0)
    if var4.get() ==1:
        my_check4.config(bootstyle="danger, toolbutton")
        print(var1.get(), var2.get(), var3.get(), var4.get(), var5.get())
    else:
        my_check4.config(bootstyle="primary, toolbutton")

def checker5():
    global var1, var2, var3, var4, var5
    var1.set(0)
    var2.set(0)
    var3.set(0)
    var4.set(0)
    if var5.get() ==1:
        my_check5.config(bootstyle="danger, toolbutton")
        print(var1.get(), var2.get(), var3.get(), var4.get(), var5.get())
    else:
        my_check5.config(bootstyle="primary, toolbutton")

def choose_video():
    global input_path, display_pic
    input_path = filedialog.askopenfilename(title="Open Video", filetype=(("MP4 Files",".mp4"), ("All Files", "*.*")))
    if input_path:
        cap = cv2.VideoCapture(input_path)
        # Get video source width and height
        vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_pic = ImageTk.PhotoImage(image = Image.fromarray(frame).resize((450, 350)))
                pic_label.config(image=display_pic)
            else:
                return
        else:
            return

# LABEL
my_label = tb.Label(text="Traffic Rules Violation Detection", font=("Helvetica",28), bootstyle="default")
my_label.pack(pady=50)

#Frame1
frame1 = tb.Frame(root, padding=20, bootstyle=PRIMARY)
frame1.pack(padx=30,side = "left")
#Frame2
frame2 = tb.Frame(root, padding=20, bootstyle=SECONDARY)
frame2.pack(padx=30,side = "right")

#OPEN VIDEO

open_button = tb.Button(frame2,text="Open Video",bootstyle="primary, outline", command=choose_video)
open_button.pack(pady=10)

pic_label = tb.Label(frame2, text="")
display_pic = ImageTk.PhotoImage(image = Image.open(r"C:\Users\rohan\Downloads\oVERT\Github\Home-GUI\assets\display.jpg").resize((450, 350)))
pic_label.config(image=display_pic)
pic_label.pack(padx=30,side = "bottom")

# CHECKBUTTON
frame3 = tb.Frame(frame1, bootstyle=PRIMARY)
frame3.pack()
var1 = IntVar()
my_check1 = tb.Checkbutton(frame3,text="Red Light Violation Detection",bootstyle="primary, toolbutton", variable = var1, onvalue =1, offvalue = 0, command = checker1)
my_check1.pack(pady=10,side = "right")

frame4 = tb.Frame(frame1, bootstyle=PRIMARY)
frame4.pack()
var2 = IntVar()
my_check2 =tb.Checkbutton(frame4,text="Detect Overspeeding",bootstyle="primary, toolbutton", variable = var2, onvalue =1, offvalue = 0, command = checker2)
my_check2.pack(pady=10,side = "right")

frame5 = tb.Frame(frame1, bootstyle=PRIMARY)
frame5.pack()
var3 = IntVar()
my_check3 =tb.Checkbutton(frame5,text="Lane Violation",bootstyle="primary, toolbutton", variable = var3, onvalue =1, offvalue = 0, command = checker3)
my_check3.pack(pady=10,side = "right")

frame6 = tb.Frame(frame1, bootstyle=PRIMARY)
frame6.pack()
var4 = IntVar()
my_check4 =tb.Checkbutton(frame6,text="Helmet Violation",bootstyle="primary, toolbutton", variable = var4, onvalue =1, offvalue = 0, command = checker4)
my_check4.pack(pady=10,side = "right")

frame7 = tb.Frame(frame1, bootstyle=PRIMARY)
frame7.pack()
var5 = IntVar()
my_check5 =tb.Checkbutton(frame7,text="Automatic Number Plate Recognition",bootstyle="primary, toolbutton", variable = var5, onvalue =1, offvalue = 0, command = checker5)
my_check5.pack(pady=10,side = "right")


# BUTTON
my_button = tb.Button(frame1,text="Start",bootstyle="success, outline", command=changer)
my_button.pack(pady=20)

root.mainloop()