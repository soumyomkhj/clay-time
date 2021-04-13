import numpy as np
import cv2
import os
import pyaudio
import math

def image2bin(image):
	imgGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_, thrash = cv2.threshold(imgGrey, 80, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	width, height = image.shape[:2]
	print ("image height",width, height)
	for contour in contours:
		approx = cv2.approxPolyDP(contour, 0.001* cv2.arcLength(contour, True), True)
		x1 ,y1, w, h = cv2.boundingRect(approx)
		print ("object w,h",w,h)
		if w+10<width:
			cv2.drawContours(image, [approx], 0, (127, 209, 28), 2)
	cv2.imshow("Output",image)


img = cv2.imread("Images/sample_mask.jpg", cv2.IMREAD_COLOR)
cv2.imshow("Input",img)
image2bin(img)
cv2.waitKey(0)
cv2.destroyAllWindows()