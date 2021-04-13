import pygame
def play_music(music_file):

	clock = pygame.time.Clock()
	try:
		pygame.mixer.music.load(music_file)
		print("Music file %s loaded!" % music_file)
	except pygame.error:
		print("File %s not found! (%s)" % (music_file, pygame.get_error()))
		return
	pygame.mixer.music.play()
	while pygame.mixer.music.get_busy():
		# check if playback has finished
		clock.tick(30)
# pick a midi music file you have ...
# (if not in working folder use full path)

import os
def music_files():
	music_dir = "MIDI/"
	midi_files = os.listdir(music_dir)
	for one in midi_files:
		yield music_dir + one


freq = 44100    # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2    # 1 is mono, 2 is stereo
buffer = 1024    # number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)
# optional volume 0 to 1.0
pygame.mixer.music.set_volume(0.8)
for music_file in music_files():
	try:
		play_music(music_file)
	except KeyboardInterrupt:
		# if user hits Ctrl/C then exit
		# (works only in console mode)
		while True:
			action = input('Enter Q to Quit, Enter to Skip. ').lower()
			if action == 'q':
				pygame.mixer.music.fadeout(1000)
				pygame.mixer.music.stop()
				raise SystemExit
			else:
				break