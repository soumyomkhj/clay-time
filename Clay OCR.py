import time

#py tesseract (also works with Raspbery Py)
import pytesseract
#Open CV (also works with Raspbery Py)
import cv2

import os
from PIL import Image, ImageEnhance, ImageFilter
tessdata_dir_config = '--psm 6 --tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
# Example config: '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
# It's important to include double quotes around the dir path.
camera_port = 1
camera = cv2.VideoCapture(camera_port)
time.sleep(0.1)  

if not os.path.exists('Images/image_frames'):
	os.makedirs('Images/image_frames')

index=0
while camera.isOpened():
	ret,frame=camera.read()
	index=index+1
	if ((index%1)==0):
		imgH, imgW, _ = frame.shape
		x1,y1,w1,h1=0,0,imgH,imgW
		text = pytesseract.image_to_string(frame, lang = 'eng')
		imageBoxes = pytesseract.image_to_boxes(frame)
		for boxes in imageBoxes.splitlines():
			boxes= boxes.split(' ')
			x,y,w,h=int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
			cv2.rectangle(frame,(x,imgH-y),(w, imgH-h),(0,255,0),3)	


		if (text.find('CAT') != -1):
			text= "Yayyyy its a CAT!"
		

		cv2.putText(frame,text, (x1+int(w1/50),y1+int(h1/50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
		font=cv2.FONT_HERSHEY_SIMPLEX
		cv2.imshow('OCR', frame)
		if cv2.waitKey(2) & 0xFF == ord('q'):
			break
	#name = './Images/image_frames' + str(index) + '.png'
	#print ('frames')
	# cv2.imwrite(name,frame)
	
	#if cv2.waitkey(10)& 0xff == ord('q'):
	#	break

camera.release()
cv2.destroyAllWindows()