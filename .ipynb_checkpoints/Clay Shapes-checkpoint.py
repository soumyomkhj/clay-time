import time
import cv2
import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
tessdata_dir_config = '--psm 6 --tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
# Example config: '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
# It's important to include double quotes around the dir path.
camera_port = input("Enter your cam port (default = 0): ")

camera = cv2.VideoCapture(int(camera_port))
time.sleep(0.1)  

index=0
while camera.isOpened():
	ret,frame=camera.read()
	index=index+1
	if ((index%5)==0):
		imgH, imgW, _ = frame.shape
		x1,y1,w1,h1=0,0,imgH,imgW

		imgGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		_, thrash = cv2.threshold(imgGrey, 80, 255, cv2.THRESH_BINARY)
		contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		for contour in contours:
			approx = cv2.approxPolyDP(contour, 0.03* cv2.arcLength(contour, True), True)
			x = approx.ravel()[0]
			y = approx.ravel()[1] - 5
			x1 ,y1, w, h = cv2.boundingRect(approx)
			if w > 20 and h> 20 and w<500:
				cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)
				if len(approx) == 3:
					cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
				elif len(approx) == 4:
					aspectRatio = float(w)/h
					#print(w)
					if aspectRatio >= 0.95 and aspectRatio <= 1.05:
					  cv2.putText(frame, "square", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
					else:
					  cv2.putText(frame, "rectangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
				elif len(approx) == 5:
					cv2.putText(frame, "Pentagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
				elif len(approx) == 6:
					cv2.putText(frame, "Hexagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
				elif len(approx) == 10:
					cv2.putText(frame, "Star", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
				else:
					cv2.putText(frame, "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
		cv2.imshow('Shapes', frame)
		#cv2.imshow('Thrash', thrash)
		if cv2.waitKey(2) & 0xFF == ord('q'):
			break
	#name = './Images/image_frames' + str(index) + '.png'
	#print ('frames')
	# cv2.imwrite(name,frame)
	
	#if cv2.waitkey(10)& 0xff == ord('q'):
	#	break

camera.release()
cv2.destroyAllWindows()