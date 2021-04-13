import time
import cv2
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import music21
from music21 import *
import math

# make input 720p
def make_640p(cap):
	cap.set(CV_CAP_PROP_FRAME_WIDTH,640)
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,480)
	return cap


# create single tone
def createTone(f,duration):
	if f>=0 and f<37:
		n1=note.Note('C3',quarterLength=duration)
	if f>=37 and f<37*2:
		n1=note.Note('C#3',quarterLength=duration)
	if f>=37*2 and f<37*3:
		n1=note.Note('D3',quarterLength=duration)
	if f>=37*3 and f<37*4:
		n1=note.Note('D#3',quarterLength=duration)
	if f>=37*4 and f<37*5:
		n1=note.Note('E3',quarterLength=duration)
	if f>=37*5 and f<37*6:
		n1=note.Note('F3',quarterLength=duration)
	if f>=37*6 and f<37*7:
		n1=note.Note('F#3',quarterLength=duration)
	if f>=37*7 and f<37*8:
		n1=note.Note('G3',quarterLength=duration)
	if f>=37*8 and f<37*9:
		n1=note.Note('G#3',quarterLength=duration)
	if f>=37*9 and f<37*10:
		n1=note.Note('A3',quarterLength=duration)
	if f>=37*10 and f<37*11:
		n1=note.Note('A#3',quarterLength=duration)
	if f>=37*11 and f<37*12:
		n1=note.Note('B3',quarterLength=duration)
	if f>=37*12 and f<480:
		n1=note.Note('C4',quarterLength=duration)
	return n1



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
	lower_red = np.array([0,80,80])
	upper_red = np.array([1,255,255])
	mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

	# upper mask (170-180)
	lower_red = np.array([170,90,90])
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

def play_music(music_file):

	clock = pygame.time.Clock()
	try:
		pygame.mixer.music.load(music_file)
		# print("Music file %s loaded!" % music_file)
	except pygame.error:
		# print("File %s not found! (%s)" % (music_file, pygame.get_error()))
		return
	pygame.mixer.music.play()
	while pygame.mixer.music.get_busy():
		# check if playback has finished
		clock.tick(30)
		

def main():
	camera_port = 3

	# duration=input("Input Duration in seconds")
	# input("Enter your cam port (default = 0): ")

	
	# turn on camera
	camera = cv2.VideoCapture(int(camera_port), cv2.CAP_DSHOW)
	cv2.waitKey(100)

	# make_640p(camera)
	width, height = 640,480


	index=0
	# testing
	
	while camera.isOpened():
		#read from camera
		_ ,frame= camera.read()
		# frame = cv2.imread("Images/sample.png", cv2.IMREAD_COLOR)
		index += 1
	
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
		

		# detect bounding box
		obj_pos,count =object_position(contours)			
		print("f\t duration")

		# scan each object to get music
		samples=stream.Stream()
		for shape_no in range(count):
			duration=obj_pos[shape_no][2]/620*8
			f=480-(obj_pos[shape_no][1]+(obj_pos[shape_no][3]/2))
			print(f,"\t",duration)
			#get midi
			notes=createTone(f,duration)
			#list of all the midi bits
			samples.append([notes])
		music_file=samples.write('midi',fp='MIDI/2.mid')
		
		freq = 44100    # audio CD quality
		bitsize = -16   # unsigned 16 bit
		channels = 2    # 1 is mono, 2 is stereo
		buffer = 1024    # number of samples
		pygame.mixer.init(freq, bitsize, channels, buffer)


		# optional master volume 0 to 1.0
		pygame.mixer.music.set_volume(0.8)
		try:
			play_music(music_file)
		except KeyboardInterrupt:
			# if user hits Ctrl/C then exit
			# (works only in console mode)
			while True:
				action = input('Enter Q to Quit, Enter to Skip. ').lower()
				if action == 'q' or action == 'Q':
					pygame.mixer.music.fadeout(1000)
					pygame.mixer.music.stop()
					raise SystemExit
				else:
					break
		# break
		cv2.waitKey(1000)
		
		
	camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()