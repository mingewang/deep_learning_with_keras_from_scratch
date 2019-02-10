# apt-get install opencv-python

import cv2
import numpy as np

#Create MSER object
# for opencv3
#mser = cv2.MSER_create()
# opencv 2
mser = cv2.MSER()

#Your image path i-e receipt path
img = cv2.imread('receipt.jpg')

#Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

vis = img.copy()

#detect regions in gray scale image
#regions, _ = mser.detectRegions(gray)
#for cv2
regions = mser.detect(gray)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(vis, hulls, 1, (0, 255, 0))

cv2.imshow('img', vis)

#for i, contour in enumerate(hulls):
#    x,y,w,h = cv2.boundingRect(contour)
#    cv2.imwrite('{}.png'.format(i), img[y:y+h,x:x+w])

cv2.waitKey(0)

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
mask = cv2.dilate(mask, np.ones((150, 150), np.uint8))

for contour in hulls:
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("text only", text_only)

for i, contour in enumerate(hulls):
    x,y,w,h = cv2.boundingRect(contour)
    cv2.imwrite('text_{}.png'.format(i), text_only[y:y+h,x:x+w])



cv2.waitKey(0)
