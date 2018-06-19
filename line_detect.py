# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import time
import cv2

import imageProcessing

bluePlayerColorLower = (106, 120, 60)
bluePlayerColorUpper = (116, 190, 150)

redPlayerColorLower = (0, 110, 65)
redPlayerColorUpper = (5, 195, 170)

blackColorLower = (0, 0, 0)
blackColorUpper = (179, 60, 50)

def getFoosballPlayersAndLineMask(hsvFrame):
    mask = cv2.inRange(hsvFrame, redPlayerColorLower, redPlayerColorUpper)
    if (cv2.countNonZero(mask) != 0):
        mask = mask + cv2.inRange(hsvFrame, (175, 110, 65), (179, 195, 170))
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=8)

    mask2 = cv2.inRange(hsvFrame, bluePlayerColorLower, bluePlayerColorUpper)
    if (cv2.countNonZero(mask2) != 0):
        mask2 = cv2.erode(mask2, None, iterations=1)
        mask2 = cv2.dilate(mask2, None, iterations=8)

    mask3 = cv2.inRange(hsvFrame, blackColorLower, blackColorUpper)
    if (cv2.countNonZero(mask3) != 0):
        mask3 = cv2.erode(mask3, None, iterations=1)
        mask3 = cv2.dilate(mask3, None, iterations=10)

    playersMask = mask + mask2 + mask3

    for linePoints in playerLines:
		cv2.line(playersMask, (linePoints[0], linePoints[1]), (linePoints[2], linePoints[3]), (256, 256, 256), 12)

    return playersMask


camera = cv2.VideoCapture("video/randomPlay.avi")

(grabbed, frame) = camera.read() 
(grabbed, frame) = camera.read() 
(grabbed, frame) = camera.read() 
(grabbed, frame) = camera.read() 
(grabbed, frame) = camera.read() 

redPlayerLines = []
bluePlayerLines = []

startFrameGrey = None 
startFramePlayerMask = None

#Detect player lines
while True:
    (grabbed, frame) = camera.read() 
	
    processedFrame = frame.copy()
    originalFrame = frame.copy()

    edgesImage = imageProcessing.getFoosballImageEdges(frame)
    playerLines = imageProcessing.getFoosballPlayerLines(edgesImage)

    if len(playerLines) == 8:
        redPlayerLines = [playerLines[0], playerLines[1], playerLines[3], playerLines[5]]
        bluePlayerLines = [playerLines[2], playerLines[4], playerLines[6], playerLines[7]]

        startFrameGrey = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2GRAY)
        startFramePlayerMask = getFoosballPlayersAndLineMask(cv2.cvtColor(originalFrame, cv2.COLOR_BGR2HSV))

        break


(grabbed, frame) = camera.read() 
lastFrame = frame
previousPlayersMask = cv2.cvtColor(frame.copy() * 0, cv2.COLOR_BGR2GRAY)


totalFrames = 0
detectedFrames = 0

while True:
    totalFrames += 1

    (grabbed, frame) = camera.read() 
	
    processedFrame = frame.copy()
    originalFrame = frame.copy()

	#Blue and Red lines
    for index, linePoints in enumerate(redPlayerLines):
        cv2.line(processedFrame, (linePoints[0], linePoints[1]), (linePoints[2], linePoints[3]), (0, 0, 256), 2)

	for index, linePoints in enumerate(bluePlayerLines):
		cv2.line(processedFrame, (linePoints[0], linePoints[1]), (linePoints[2], linePoints[3]), (256, 0, 0), 2)

	#Blue and Red Player marks
    blurredFrame = cv2.GaussianBlur(originalFrame.copy(), (5, 5), 0)
    hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)
	
    centroids = imageProcessing.getFoosballPlayerPositionOnLines(redPlayerLines, hsvFrame, redPlayerColorLower, redPlayerColorUpper, playerDetectionLineWidth = 10)

    for c in centroids:
        cv2.circle(processedFrame, c, 8, (0, 0, 256), 2)

    centroids = imageProcessing.getFoosballPlayerPositionOnLines(bluePlayerLines, hsvFrame, bluePlayerColorLower, bluePlayerColorUpper, playerDetectionLineWidth = 10)

    for c in centroids:
        cv2.circle(processedFrame, c, 8, (256, 0, 0), 2)

    playersMask = getFoosballPlayersAndLineMask(hsvFrame)

    cv2.imshow("Mask", playersMask)

    lastFrameGrey = cv2.cvtColor(lastFrame, cv2.COLOR_BGR2GRAY)
    currentFrameGrey = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2GRAY)
    differenceFrame = cv2.absdiff(lastFrameGrey, currentFrameGrey)
    #differenceFrame = cv2.absdiff(startFrameGrey, currentFrameGrey)
    cv2.imshow("DifferenceRaw", differenceFrame)


    differenceFrame = cv2.GaussianBlur(differenceFrame, (5, 5), 0)
    ret, differenceFrame = cv2.threshold(differenceFrame, 15, 255, cv2.THRESH_BINARY)

    differenceFrame -= previousPlayersMask
    #differenceFrame -= startFramePlayerMask
    differenceFrame -= playersMask

    differenceFrame = cv2.erode(differenceFrame, None, iterations=1)
    differenceFrame = cv2.dilate(differenceFrame, None, iterations=3)

    ret, differenceFrame = cv2.threshold(differenceFrame, 25, 255, cv2.THRESH_BINARY)

    #[y1:y2, x1:x2]
    differenceFrame = differenceFrame[80:400, 0:640]

    center, x, y, radius, cnts = imageProcessing.findCenterOfLargestContour(differenceFrame)

    if radius > 3 and center and x and y:
        y += 80
		# draw the circle and centroid on the frame,
		# then update the list of tracked points
        #cv2.circle(processedFrame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(processedFrame, (center[0], center[1] + 80), 5, (0, 255, 255), -2)
        detectedFrames += 1

    print(float(detectedFrames) / float(totalFrames))
    #cv2.drawContours(processedFrame, cnts, -1, (0,255,0), 3)

    cv2.imshow("Difference", differenceFrame)

    previousPlayersMask = playersMask

    cv2.imshow("Hough", processedFrame)

    lastFrame = originalFrame

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
