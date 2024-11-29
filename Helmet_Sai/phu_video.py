# phu_video.py --catching 1 --model C:\Users\rohan\Downloads\oVERT\Github\Helmet-Sai\biker_yolov5s.pt --video "C:\Users\rohan\Downloads\1.mp4"  


from . import phu_yolov5
import sys, getopt
import torch
import cv2, time
import pafy
import base64
from easyocr import Reader
reader = Reader(['en'])
model_plates = torch.hub.load(r"yolov5", 'custom', path=r"yolov5/best_number_plate.pt", source='local') # ADDED

def main(argv, stframe, cropimage, plateimage,image_files):
	# arg check
	print(argv.source)
	# opts, args = getopt.getopt( argv, "yvcm" , ["youtube=", "video=", "catching=","model="] )
	

	# catching = False
	# source = None
	# model_path = "biker_yolov5s.pt"
	# for opt, arg in opts:
	# 	if opt in ("-y","--youtube"):
	# 		video = pafy.new( arg )
	# 		best = video.getbest( preftype="mp4" )
	# 		source = best.url
	# 	elif opt in ("-v","--video"):
	# 		source = arg
	# 	elif opt in ("-c","--catching"):
	# 		if arg == "1":
	# 			catching = True
	# 		else :
	# 			catching = False
	# 	elif opt in ("-m","--model"):
	# 		model_path = arg

	# if source == None :
	# 	source = 0 

	catching = argv.catching
	source = argv.source
	model_path = argv.model
	print( "using source :", source )
	print( "using model : "+model_path)
	
	cap = cv2.VideoCapture( source )

	model = torch.hub.load( "./yolov5", "custom", model_path, source="local")
	detective = phu_yolov5.Detective( model, catching=catching )

	FPS = 15
	SPF = 1/FPS
	image_html = ""
	while 1:
		start_time = time.time()
		ret, frame = cap.read()
		if not ret:
			break
		boxes, biker_which_has_nohel = detective.detect( frame )
		img = phu_yolov5.draw_boxes( frame, boxes)

		# cv2.imshow( "asfgas", img )
		# cv2.imwrite(r"C:\Users\rohan\Downloads\oVERT\Github\Home-GUI\static\img\output.jpg",img)
		stframe.image(img, channels="BGR", use_column_width=True)
		de = [box.astype(int) for box in biker_which_has_nohel]
		for box in de:
			# print(box)
			# if box[5]>=1:
			crop_img = img[box[1]:box[3], box[0]:box[2]]
			cropimage.image(crop_img, channels="BGR",  width = 300)
			
			results = model_plates(crop_img) #ADDED
			if 0 in results.pandas().xyxy[0]['class'] and results.pandas().xyxy[0]['confidence'].values[0]>=0.3:
					crop = results.crop(save=False)[0]["im"] 
					plateimage.image(crop, channels="BGR",  width = 300)
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
					detection = reader.readtext(crop)
					if len(detection)!=0 and detection[0][2]>=0.3:
						text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
						print(text)

		if cv2.waitKey(1) == ord( "q" ):
			break

		run_time = time.time() - start_time

		if source != 0:
			if run_time < SPF:
				time.sleep( SPF - run_time )
			else:
				passframe = round( run_time/SPF ) - 1
				current_frame = cap.get( cv2.CAP_PROP_POS_FRAMES )
				cap.set( cv2.CAP_PROP_POS_FRAMES, current_frame + passframe )

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main( sys.argv[1:] )


