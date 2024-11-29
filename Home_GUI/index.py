from flask import Flask, render_template, request, flash, redirect, Response
from werkzeug.utils import secure_filename
import os
import sys
import time
from flask_socketio import SocketIO
import threading

# sys.path.insert(1, 'RLVD_pavan')
# import rlvd_track_pavan as rlvd
# outputFrame = None
# lock = threading.Lock()
# time.sleep(2.0)
app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = "Home-GUI/static/vid"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
import json
from PIL import Image
import glob
socket = SocketIO(app)
# socket = SocketIO(app,logger=True, engineio_logger=True)


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            print("hue1")
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print("hue2")
            flash('No selected file')
            return redirect(request.url)
        if file:
            print("hue3")
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filename = 'Home-GUI/static/vid/' + filename
            print(request.form)
            if "rlvd" in request.form:
                print("Doing RLVD...")
                os.system(rf'python C:\Users\rohan\Downloads\oVERT\Github\RLVD_pavan\rlvd_track_pavan.py --source {filename} --show-vid --yolo-weights C:\Users\rohan\Downloads\oVERT\Github\Helmet-sai\yolov5s.pt --save-crop --save-vid')
                # rlvd.run(filename)
            if "speed" in request.form:
                print("Doing Speed Violation...")
                os.system(rf'python C:\Users\rohan\Downloads\oVERT\Github\SpeedTracking-rohan\track.py --source {filename} --yolo-weights C:\Users\rohan\Downloads\oVERT\Github\SpeedTracking-rohan\yolov5s.pt --classes 2 --save-crop --show-vid')
            if "helmet" in request.form:
                print("Doing Helmet Violation...")
                # os.system(rf'python C:\Users\rohan\Downloads\oVERT\Github\Helmet-sai\Yolov5_DeepSort_Pytorch\track.py --source {filename} --yolo-weights "C:\Users\rohan\Downloads\oVERT\Github\Helmet-sai\biker_yolov5m.pt" --img 640  --show-vid --save-crop --save-vid --conf-thres 0.80')
                os.system(rf'python phu_video.py --catching 1 --model C:\Users\rohan\Downloads\oVERT\Github\Helmet-Sai\biker_yolov5s.pt --video {filename}')
            if "lane" in request.form:
                print("Doing Lane Detection...")
                os.system(rf'python LaneViolation-rohan\finalPro.py -i {filename}')
            if "anpr" in request.form:
                print("Doing ANPR...")
                os.system(rf'python Helmet-sai\detect.py --weights C:\Users\rohan\Downloads\oVERT\Github\ANPR-rohan\new_weights.py\best.pt --img 416 --conf 0.4 --source {filename} --save-crop --view-img')
        # return render_template("index.html",  image_list=image_list)

    return render_template("index.html")

# , image_list=json.dumps(image_list)
@socket.on('message')
def handlemsg(msg):
    image_list = []
    for filename in glob.glob(r'Home-GUI\static\img\plates\*jpg'): #assuming jpg
        image_list.append(filename.replace("Home-GUI",""))
    socket.send(image_list)


if __name__ == "__main__":
    # app.run(debug=True)
    socket.run(app)

# https://www.shanelynn.ie/asynchronous-updates-to-a-webpage-with-flask-and-socket-io/