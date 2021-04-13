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


# create single pixel tone
def createTone(volume,f,duration):
	fs=44100
	volume = math.log(float(volume)) / math.log(100)
	# create sine wave
	samples = volume*(np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
	return samples


#Get the contours
def image2bin(image):

	# convert to binary
	_, thrash = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)

	kernel = np.ones((7,7),np.uint8)
	thrash = cv2.morphologyEx(thrash, cv2.MORPH_OPEN, kernel)
	thrash = cv2.morphologyEx(thrash, cv2.MORPH_CLOSE, kernel)

	# get contours
	contours, _ = cv2.findContours(thrash.astype(np.uint8).copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours, thrash


# Get the masked image
def mask_red(frame):
	img_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	# HSV

	# lower mask (0-10)
	lower_red = np.array([0,50,50])
	upper_red = np.array([10,255,255])
	mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

	# upper mask (170-180)
	lower_red = np.array([170,50,50])
	upper_red = np.array([180,255,255])
	mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

	# join my masks
	mask = mask0+mask1
	mask = cv2.GaussianBlur(mask,(7,7),0)
	return mask


#get the bounding box
def object_position(contours):
	count=0
	obj_pos=[]
	for contour in contours:
		approx = cv2.approxPolyDP(contour, 0.001* cv2.arcLength(contour, True), True)
		x1 ,y1, w, h = cv2.boundingRect(approx)
		obj_pos.append([x1 ,y1, w, h])
		count += 1
	# print(obj_pos,count)

	# sorted according to starting points
	obj_pos.sort(key=lambda x: x[0])
	# print(obj_pos,count)

	return obj_pos,count


#get the cross section
def find_limits(conture,x_pos,y,h):
	bottom=-1
	top=-1
	for y_pos in range(h):
		if y+y_pos == 480:
			y_pos-=1
		if conture[y+y_pos,x_pos]==255:
			top=y+y_pos
			break
	for y_pos_r in range(h):
		if conture[y+h-y_pos_r,x_pos]==255:
			bottom=y+h-y_pos_r
			break
	return bottom,top


#class to store tunes
class clay_shape(object):
	"""docstring for clay_shape"""
	def __init__(self,sample,x1):
		# self.shape_no= n
		self.sample=sample
		self.x1=x1
		# self.y1=y1
		# self.w=w
		# self.h=h

		
#the final tune
def play_tone(samples,count):
	PyAudio = pyaudio.PyAudio
	p = PyAudio()
	fs=44100
	stream=[]

	# init streams
	for x in range(count):
		stream.append(p.open(format=pyaudio.paFloat32,channels=1,rate=fs,output=True))
		# wait
	
	#start streams
	for x in range(count):
		stream[x].write(samples[x].sample.tobytes())

	#close streams
	for x in range(count):
		stream[x].stop_stream()
		stream[x].close()
	
	p.terminate()



def main():
	camera_port = 3
	duration = 2

	# duration=input("Input Duration in seconds")
	# input("Enter your cam port (default = 0): ")

	
	# turn on camera
	camera = cv2.VideoCapture(int(camera_port), cv2.CAP_DSHOW)
	cv2.waitKey(100)
	# make_640p(camera)
	width, height = 640,480

	duration = duration/width # (per note)

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
			contours, thrash=image2bin(mask)
			cv2.imshow("Trash_Shapes", thrash)
			# break	
			if cv2.waitKey(2) & 0xFF == ord('q'):
				break
			frame_copy=thrash.copy()

			# detect bounding box
			obj_pos,count =object_position(contours)			

			# scan each object to get music
			samples=[]
			for shape_no in range(count):
				sample=[]
				# scan  left to right along x axis
				for x_pos in range(obj_pos[shape_no][0],obj_pos[shape_no][0]+obj_pos[shape_no][2]):

					# scan vertical line for start and end
					bottom,top= find_limits(thrash,x_pos,obj_pos[shape_no][1],obj_pos[shape_no][3])
					# print (bottom,top)
					
					# set volume (max volume=height/3)
					if bottom-top>(height/3):
						volume=100
					else:
						volume=(bottom-top)*100/(height/3)

					# set frequency from convert 0-480p scale to 210hz-510hz
					f=(bottom+top)/2
					f=(f*1000/480)

					# bring in the tune data
					# print(shape_no)
					sample=np.append(sample,createTone(volume,f,duration))
					sample=np.array(sample)
				
				# when is the next one?
				if shape_no+1 !=count and obj_pos[shape_no+1][0]>obj_pos[shape_no][0]+obj_pos[shape_no][2]:
					# after this one ends -> wait
					next_shape = obj_pos[shape_no+1][0]-obj_pos[shape_no][0]+obj_pos[shape_no][2]
					# print(next_shape)
					time.sleep(next_shape*duration)
				elif shape_no+1 !=count and obj_pos[shape_no+1][0]<obj_pos[shape_no][0]+obj_pos[shape_no][2]:
					# before this one ends -> play togather
					next_shape = obj_pos[shape_no+1][0]-obj_pos[shape_no][0]+obj_pos[shape_no][2]
					next_shape = -next_shape
					# print(next_shape)

				#list of all the 
				samples.append(clay_shape(sample,obj_pos[shape_no][0]))
				
				# PyAudio = pyaudio.PyAudio
				# p = PyAudio()
				# fs=44100
				# # for paFloat32 sample values must be in range [-1.0, 1.0]
				# stream = p.open(format=pyaudio.paFloat32,
				# 				channels=1,
				# 				rate=fs,
				# 				output=True)
				# stream.write(samples[shape_no].sample.tobytes())
				# stream.stop_stream()
				# stream.close()
				# p.terminate()

			play_tone(samples,count)
			# innitalize all the streams of audio from different clays
			# for i_pos in range(count):
				# obj_tune

			del obj_pos
			cv2.waitKey(1000)
			
		
	camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()