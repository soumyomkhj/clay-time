import time
import cv2
import numpy as np
import os
import pyaudio
import math
from PIL import Image, ImageEnhance, ImageFilter



# play single tone
def playTone(volume,f,duration):
	fs=44100
	#innit pyaudio
	PyAudio = pyaudio.PyAudio

	# create sine wave
	samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
	
	p = PyAudio()
	
	# for paFloat32 sample values must be in range [-1.0, 1.0]
	stream = p.open(format=pyaudio.paFloat32,
					channels=1,
					rate=fs,
					output=True)
	
	# stream = p.open(format = p.get_format_from_width(1),channels = 2,rate = sps,output = True)
	# print(samples)
	stream.write((samples*volume).tobytes())
	stream.stop_stream()
	stream.close()
	p.terminate()





#Get the contures
def image2bin(image):

	# convert to binary
	_, thrash = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
	# get contures
	contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	width, height = image.shape[:2]
	
	blank_img = np.zeros([width, height],np.float32)
	
	# print ("image height",width, height)
	approx_shapes = []
	for contour in contours:
		# create poli lines
		approx = cv2.approxPolyDP(contour, 0.001* cv2.arcLength(contour, True), True)
		x1 ,y1, w, h = cv2.boundingRect(approx)
		no_obj=0
		if w+10<width and w>10 and h>10:
			# for all accepable poli lines
			# print ("object w,h",w,h)
			approx_shapes.append(approx)
			cv2.fillPoly(blank_img, [approx], 255)
			cv2.drawContours(image, [approx], 0, (127, 209, 28), 2)
			no_obj += 1
	# print (approx_shapes)
	approx_shapes = np.array(approx_shapes)
	return approx_shapes, blank_img





# Get the masked image
def mask_red(frame):
	hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	# HSV
	low_red = np.array([140,40,20])
	high_red = np.array([190,255,255])
	#masking
	mask= cv2.inRange(hsv_frame,low_red,high_red)
	mask=cv2.bitwise_not(mask)
	return mask





def main():
	camera_port = 3
	# input("Enter your cam port (default = 0): ")

	# turn on camera
	camera = cv2.VideoCapture(int(camera_port), cv2.CAP_DSHOW)
	time.sleep(0.1)  

	index=0
	while camera.isOpened():
		_ ,frame=camera.read()
		width, height = frame.shape[:2]
		index += 1
		# every 10th frame
		if ((index%10)==0):

			# Get the masked image
			mask=mask_red(frame)
			# print ("frame no", index,"\n")
			
			#Get the contures
			clays_data,conture=image2bin(mask)
			
			
			# print (clays_data.shape)

			# for every shape 
			for clay in clays_data:
				x1 ,y1, w, h = cv2.boundingRect(clay)
				# print (x1 ,y1, w, h,"\n")
				

				blank_img=np.zeros([width, height],np.float32)
				cv2.fillPoly(blank_img, [clay], 255)
				# cv2.imshow('Clays', blank_img)
			
				playTone(h/200,height-y1,.5)

				# print (clay.shape)

			# Show image
			cv2.imshow('Original', frame)
			cv2.imshow('Shapes', conture)
			# break
			if cv2.waitKey(2) & 0xFF == ord('q'):
				break
		
	camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()