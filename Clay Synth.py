import numpy as np
import cv2
import os
import pyaudio
import math
# Sample per sec / Frequency / Duration (s)
def playTone(volume,fs,f,duration):
	PyAudio = pyaudio.PyAudio
	volume = math.log(float(volume)) / math.log(100)
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

while True:
	volume = 0.5     # range [0.0, 1.0]
	sps = 44100       # sampling rate, Hz, must be integer
	freq = 440.0        # sine frequency, Hz, may be float
	dur = .4			# in seconds, may be float

	volume = input("Enter volume 1-100: ")
	# playTone(float(volume), sps,float(freq),dur)
	# Sa 240 Hz, Re 270 Hz, Ga 300 Hz, Ma 320 Hz, Pa 360 Hz, Dha 400 Hz, and Ni 450 Hz, Sa 480 Hz
	playTone(volume, sps,float(240),dur)
	playTone(volume, sps,float(270),dur)
	playTone(volume, sps,float(300),dur)
	playTone(volume, sps,float(320),dur)
	playTone(volume, sps,float(360),dur)
	playTone(volume, sps,float(400),dur)
	playTone(volume, sps,float(450),dur)
	playTone(volume, sps,float(480),dur)

	playTone(volume, sps,float(480),dur)
	playTone(volume, sps,float(450),dur)
	playTone(volume, sps,float(400),dur)
	playTone(volume, sps,float(360),dur)
	playTone(volume, sps,float(320),dur)
	playTone(volume, sps,float(300),dur)
	playTone(volume, sps,float(270),dur)
	playTone(volume, sps,float(240),dur)
	pass