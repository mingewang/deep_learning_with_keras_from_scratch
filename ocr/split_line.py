#!/usr/bin/python
# copied from:
# https://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document
# with minor changes
# you may need to install opencv and its python libs
# apt-get install opencv-python

import cv2
import numpy as np

## (1) read
img = cv2.imread("ocr_test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## (2) threshold
th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

## (3) minAreaRect on the nozeros
pts = cv2.findNonZero(threshed)
ret = cv2.minAreaRect(pts)

(cx,cy), (w,h), ang = ret
if w>h:
    w,h = h,w
    ang += 90

## (4) Find rotated matrix, do rotation
M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))

## (5) find and draw the upper and lower boundary of each lines
# for python3
#hist = cv2.reduce(rotated,1, cv2.cv.REDUCE_AVG).reshape(-1)
# for python2
hist = cv2.reduce(rotated,1, cv2.cv.CV_REDUCE_AVG).reshape(-1)

th = 2
H,W = img.shape[:2]
uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

rotated2 = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
for y in uppers:
    cv2.line(rotated2, (0,y), (W, y), (255,0,0), 1)
    print("uppers y is:", y )

for y in lowers:
    cv2.line(rotated2, (0,y), (W, y), (0,255,0), 1)
    print("lowers y is:", y )

cv2.imwrite("result.png", rotated2)

# write file line by line
i = 0
for ty in uppers:
  found_top = False
  for ly in lowers:
    if found_top == False and ly > ty:
      found_top = True
      print("ly is:", ly )
      print("ty is:", ty )
      # revert black/write for our ocr format
      roi = cv2.bitwise_not( rotated[ty-2:ly+2, 0:W] )
      cv2.imwrite('line_{}.png'.format(i), roi)
      i = i+1
      
  

