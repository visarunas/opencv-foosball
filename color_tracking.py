# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import time
import cv2
import os

import imageProcessing
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())


ballLower = (12, 100, 70)
ballUpper = (20, 140, 220)

print ballLower
print ballUpper
#greenLower = (140, 95, 35)
#greenUpper = (220, 170, 60)

pts = deque(maxlen=args["buffer"])
 
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture("video/randomPlay.avi") 
# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])
	
if not camera.isOpened():
	print "Camera not opened"
	exit()
	
(grabbed, frame) = camera.read() 
(grabbed, frame) = camera.read() 
(grabbed, frame) = camera.read() 
(grabbed, frame) = camera.read() 
(grabbed, frame) = camera.read() 

totalFrames = 0
detectedFrames = 0

# keep looping
while True:
	totalFrames += 1

	start = time.time()

	(grabbed, frame) = camera.read()
 
	if not grabbed:
		print "End of video"
		break
 
	frame = cv2.GaussianBlur(frame, (5, 5), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(hsv, ballLower, ballUpper)
	mask = cv2.erode(mask, None, iterations=1)
	mask = cv2.dilate(mask, None, iterations=2)

	center, x, y, radius, cnts = imageProcessing.findCenterOfLargestContour(mask)
	cv2.drawContours(frame, cnts, -1, (0,255,0), 3)
 
	# only proceed if the radius meets a minimum size
	if radius > 2:
		cv2.circle(frame, center, 5, (0, 0, 255), -1)
		detectedFrames += 1
 
	# update the points queue
	pts.appendleft(center)

		# loop over the set of tracked points
	for i in xrange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)

	end = time.time()

	cv2.imshow("Mask", mask)
	cv2.imshow("Original", frame)

	key = cv2.waitKey(1) & 0xFF

	print(float(detectedFrames) / float(totalFrames))
        
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
		
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
