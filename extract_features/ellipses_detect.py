import cv2
import numpy as np
import imutils

image = cv2.imread('front2.png', cv2.IMREAD_COLOR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Downsize image (by factor 4) to speed up morphological operations
gray = cv2.resize(gray, dsize=(0, 0), fx=0.25, fy=0.25)
cv2.imshow("Gray resized", gray)
cv2.imwrite('Gray_resized.png', gray)

# Morphological opening: Get rid of the stuff at the top of the ellipse
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
cv2.imshow("Gray removed noise", gray)
cv2.imwrite('Gray_removed_noise.png', gray)

# Resize image to original size
gray = cv2.resize(gray, dsize=(image.shape[1], image.shape[0]))

# Find contours
cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw found contours in input image
image = cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)

if len(cnts) != 0:
    for cont in cnts:
        if cont.size < 10 or cv2.contourArea(cont) < 100:
            continue
        elps = cv2.fitEllipse(cont)
        cv2.ellipse(image,elps,(0,255,0),2)

# Downsize image
out_image = cv2.resize(image, dsize=(0, 0), fx=0.25, fy=0.25)
cv2.imshow("Output image", out_image)
cv2.imwrite('Output_image.png', out_image)