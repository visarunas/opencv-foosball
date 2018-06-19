import cv2
import numpy as np

def findCenterOfLargestContour(mask): 
    # find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
		
	center = None
	x = None
	y = None
	radius = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		if M["m00"] != 0:
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		
	return center, x, y, radius, cnts


def getFoosballPlayerPositionOnLines(linePointsList, hsvImage, playerColorLower, playerColorUpper, playerDetectionLineWidth):
    centroids = []

    for linePoints in linePointsList: 
		#[y1:y2, x1:x2]
        regionImage = hsvImage[linePoints[1] : linePoints[3], linePoints[0] - playerDetectionLineWidth : linePoints[2] + playerDetectionLineWidth]
        mask = cv2.inRange(regionImage, playerColorLower, playerColorUpper)

        if (cv2.countNonZero(mask) != 0):
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=4)

            contours0 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            moments  = [cv2.moments(cnt) for cnt in contours0]

            for m in moments:
                if (m['m00'] != 0):
                    centroids.append( (int(m['m10'] / m['m00'] + linePoints[0] - playerDetectionLineWidth), int(m['m01'] / m['m00'] + linePoints[1])) )


    return centroids
    

def getFoosballImageEdges(image):
    low_threshold = 30
    high_threshold = 100

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, low_threshold, high_threshold, apertureSize=3)

    height, width = edges.shape

    rectanglewidth = 15
    cv2.rectangle(edges, (width / 2 + rectanglewidth, 0 + height / 4), (width / 2 -
                                                                        rectanglewidth, height - height / 4), color=(0, 0, 0), thickness=cv2.FILLED)

    cv2.rectangle(edges, (width / 2 + width / 4 + rectanglewidth, 0 + height / 4), (width / 2 +
                                                                                    width / 4 - rectanglewidth, height - height / 4), color=(0, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(edges, (width / 2 - width / 4 + rectanglewidth, 0 + height / 4), (width / 2 -
                                                                                    width / 4 - rectanglewidth, height - height / 4), color=(0, 0, 0), thickness=cv2.FILLED)

    cv2.rectangle(edges, (0 + width / 8 + rectanglewidth, 0 + height / 4), (0 + width /
                                                                            8 - rectanglewidth, height - height / 4), color=(0, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(edges, (width - width / 8 + rectanglewidth, 0 + height / 4), (width -
                                                                                width / 8 - rectanglewidth, height - height / 4), color=(0, 0, 0), thickness=cv2.FILLED)
    
    return edges

def getFoosballPlayerLines(edgeImage):
    height, width = edgeImage.shape
    linePointsList = []
    mergedLinesList = []
    linePointDifference = width / 20
    anglePrecision = 0.1

    lines = cv2.HoughLines(edgeImage, 1, 1 * np.pi/180, 130)
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if (theta + anglePrecision > 3.14 and theta - anglePrecision < 3.14) or (theta + anglePrecision > 0 and theta - anglePrecision < 0):
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    topX, topY = lineIntersection(
                        ((x1, y1), (x2, y2)), ((0, 0), (width, 0)))
                    botX, botY = lineIntersection(
                        ((x1, y1), (x2, y2)), ((0, height), (width, height)))

                    linePointsList.append(list((topX, topY, botX, botY)))

        for linePoints1 in linePointsList:
            merged = False
            for linePoints2 in mergedLinesList:
                if (
                        (linePoints2[0] - linePointDifference <= linePoints1[0] <= linePoints2[0] + linePointDifference) and
                        (linePoints2[1] - linePointDifference <= linePoints1[1] <= linePoints2[1] + linePointDifference) and
                        (linePoints2[2] - linePointDifference <= linePoints1[2] <= linePoints2[2] + linePointDifference) and
                        (linePoints2[3] - linePointDifference <= linePoints1[3]
                            <= linePoints2[3] + linePointDifference)
                ):
                    # Merge lines
                    linePoints2[0] = (linePoints1[0] + linePoints2[0]) / 2
                    linePoints2[1] = (linePoints1[1] + linePoints2[1]) / 2
                    linePoints2[2] = (linePoints1[2] + linePoints2[2]) / 2
                    linePoints2[3] = (linePoints1[3] + linePoints2[3]) / 2
                    merged = True
                    break

            if not merged:
                mergedLinesList.append(linePoints1)

        mergedLinesList.sort(key=lambda line: line[0])
        return mergedLinesList

    return []


def increaseSaturation(hsvFrame, saturationModifier):
    hsvFrame = hsvFrame.astype("float32")

    (h, s, v) = cv2.split(hsvFrame)
    s = s * saturationModifier
    s = np.clip(s, 0, 255)
    hsv = cv2.merge([h, s, v])

    return hsv.astype("uint8")


def lineIntersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
