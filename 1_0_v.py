import time
import cv2
import numpy as np
import os
import pyaudio
import math
from PIL import Image, ImageEnhance, ImageFilter

# make input 720p
def make_640p(cap):
	cap.set(CV_CAP_PROP_FRAME_WIDTH,640)
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,480)
	return cap


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
	return samples





#Get the contours
def image2bin(image):

	# convert to binary
	_, thrash = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)

	kernel = np.ones((5,5),np.uint8)
	thrash = cv2.morphologyEx(thrash, cv2.MORPH_OPEN, kernel)
	thrash = cv2.morphologyEx(thrash, cv2.MORPH_CLOSE, kernel)
	# cv2.imshow("thrash",thrash)

	# get contours
	contours, _ = cv2.findContours(thrash.astype(np.uint8).copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours, thrash





# Get the masked image
def mask_red(frame):
	hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	# HSV
	low_red = np.array([140,80,50])
	high_red = np.array([190,255,255])
	#masking
	mask= cv2.inRange(hsv_frame,low_red,high_red)
	# mask=cv2.bitwise_not(mask)
	return mask


def obj_tune(shape,dur):
	x1 ,y1, w, h = cv2.boundingRect(clay)
	# print (x1 ,y1, w, h,"\n")
	blank_img=np.zeros([width, height],np.float32)
	cv2.fillPoly(blank_img, [clay], 255)
	# playTone(h/200,height-y1,.5)
	return sample

def object_position(contours):
	count=0
	obj_pos=[]
	for contour in contours:
		approx = cv2.approxPolyDP(contour, 0.001* cv2.arcLength(contour, True), True)
		x1 ,y1, w, h = cv2.boundingRect(approx)
		obj_pos.append([x1 ,y1, w, h])
		count += 1
	print(obj_pos,count)

	# sorted according to starting points
	obj_pos.sort(key=lambda x: x[0])
	print(obj_pos,count)

	return obj_pos,count


def find_limits(conture,copy,x_pos,y,h):
	bottom=-1
	top=-1
	print ("scanning",x_pos)
	for y_pos in range(h):

		# testing
		# copy = cv2.circle(copy, (x_pos,y+y_pos), radius=0, color=(50,50, 50), thickness=-1)
		# cv2.imshow("Scanning...",copy)
		
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
		if conture[y+y_pos,x_pos]>0:
			# print("\n found top",x_pos,y+y_pos)
			top=y+y_pos
			break
	# print ("scanning bottom") 
	for y_pos_r in range(h):
		
		# testing
		# copy = cv2.circle(copy, (x_pos,y+h-y_pos_r), radius=0, color=(100,100, 100), thickness=-1)
		# cv2.imshow("Scanning...",copy)
		
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
		if conture[y+h-y_pos_r,x_pos]==255:
			# print("\n found bottom",x_pos,y+h-y_pos_r)
			bottom=y+h-y_pos_r
			break
	return bottom,top
		# print("\n")






def main():
	camera_port = 3
	duration = 15

	# duration=input("Input Duration in seconds")
	# input("Enter your cam port (default = 0): ")

	duration = duration/640 # (per note)
	
	# turn on camera
	camera = cv2.VideoCapture(int(camera_port), cv2.CAP_DSHOW)
	time.sleep(0.1)
	# make_640p(camera)
	width, height = 640,480


	# test
	def color_pick(event,x,y,flags,params):
		frame_copy = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
		if event!=0:
			print (frame_copy[y,x])
			print (frame_copy2[y,x])



	cv2.namedWindow("color_picker")
	cv2.setMouseCallback("color_picker",color_pick)

	index=0

	# testing
	frame = cv2.imread("Images/sample.png", cv2.IMREAD_COLOR)
	
	while camera.isOpened():
		#read from camera
		# _ ,frame= camera.read()
		index += 1
		
		# every 10th frame
		if ((index%10)==0):

			cv2.imshow('Original', frame)

			# Get the masked image
			mask=mask_red(frame)
			# print ("frame no", index,"\n")
			
			#Get the contours
			frame_copy=frame.copy()
			cv2.imshow("color_picker", frame_copy)
			contours, thrash=image2bin(mask)

			# detect bounding box
			obj_pos,count =object_position(contours)			

			# scan each object
			for shape_no in range(count):

				'''
				testing
				print ("Shape: ",shape_no+1)
				print(obj_pos[shape_no])
				frame_copy=cv2.rectangle(frame_copy,(obj_pos[shape_no][0],obj_pos[shape_no][1]),(obj_pos[shape_no][0]+obj_pos[shape_no][2],obj_pos[shape_no][1]+obj_pos[shape_no][3]),(100,100,100),2)
				cv2.imshow("box Show",frame_copy)
				'''

				# scan  left to right along x axis
				for x_pos in range(obj_pos[shape_no][0],obj_pos[shape_no][0]+obj_pos[shape_no][2]):

					# scan vertical line for start and end
					bottom,top= find_limits(thrash,frame_copy,x_pos,obj_pos[shape_no][1],obj_pos[shape_no][3])
					
					# test mark top, bottom
					# frame_copy = cv2.circle(frame_copy, (x_pos,top), radius=0, color=(100, 100, 0), thickness=-1)
					# frame_copy = cv2.circle(frame_copy, (x_pos,bottom), radius=0, color=(100, 100, 0), thickness=-1)

				print ("___________________________________")

					


			# innitalize all the streams of audio from different clays
			# for i_pos in range(count):
				# obj_tune

			del obj_pos
			cv2.waitKey(1000)
			# break	
			if cv2.waitKey(2) & 0xFF == ord('q'):
				break
		
	camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()