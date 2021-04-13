import numpy as np
import pyaudio
import cv2
import os
import math        #import needed modules
# BITRATE = 5000     #number of frames per second/frameset.
# FREQUENCY = 100000     #Hz, waves per second, 261.63=C4-note.
# LENGTH = 1    #seconds to play sound

def playTone(BITRATE,FREQUENCY,LENGTH):
	PyAudio = pyaudio.PyAudio     #initialize pyaudio

	if FREQUENCY > BITRATE:
	   BITRATE = FREQUENCY+100
	NUMBEROFFRAMES = int(BITRATE * LENGTH)
	RESTFRAMES = NUMBEROFFRAMES % BITRATE
	WAVEDATA = ''

	#generating waves
	for x in range(NUMBEROFFRAMES):
		WAVEDATA = WAVEDATA+chr(int(math.sin(x/((BITRATE/FREQUENCY)/math.pi))*127+128))
	for x in range(RESTFRAMES):
		WAVEDATA = WAVEDATA+chr(128)

	#print(WAVEDATA)

	p = PyAudio()
	stream = p.open(format = p.get_format_from_width(1),channels =     2,rate = BITRATE,output = True)
	stream.write(WAVEDATA)
	stream.stop_stream()
	stream.close()
	p.terminate()


while True:
	leng=1
	bit, freq = input("Enter BITRATE, FREQUENCY: ").split()
	playTone(int(bit),int(freq),int(leng))
	pass