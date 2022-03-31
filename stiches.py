#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#cython: language_level=3, boundscheck=False
###############################################################################
#
# Playing with stiching 2 360 degree pictures
#
###############################################################################

import cv2
import numpy as np
import sys

#---------------------------------------------------------------------
def toPanorama(image):
    image_size = image.shape
    maxRadius = max(image_size[0] / 2.0, image_size[1] / 2.0)

    output_image = cv2.warpPolar(image, (-1, -1), (image_size[0] / 2.0, image_size[1] / 2.0),
                                maxRadius, cv2.WARP_POLAR_LINEAR)

    # output_image = cv2.rotate(output_image, cv2.ROTATE_90_CLOCKWISE)
    return output_image

#---------------------------------------------------------------------

def showSmallPic(name, pic):
    pic = cv2.resize(pic, (pic.shape[0] // 6, pic.shape[1] // 6))
    cv2.imshow(name, pic)

#---------------------------------------------------------------------

if len(sys.argv) < 3:
    print("Usage: python to-panorama.py <left file> <right file>\n")
    print("Too few command line arguments\n")
    sys.exit(2)

capLeft = cv2.VideoCapture(sys.argv[1])
capRight = cv2.VideoCapture(sys.argv[2])

capLeft.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capLeft.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capRight.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capRight.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

_,leftPic = capLeft.read()
_,rightPic = capRight.read()

if leftPic is None or rightPic is None:
    print("Could not read one of the pictures\n")
    sys.exit(2)

cv2.imshow("Left pic", leftPic)
cv2.imshow("Right pic", rightPic)
cv2.waitKey(0)

# leftPic = toPanorama(leftPic)
# rightPic = toPanorama(rightPic)
#
# showSmallPic("Left panorama", leftPic)
# showSmallPic("Right panorama", rightPic)

leftPicGray = cv2.cvtColor(leftPic, cv2.COLOR_BGR2GRAY)
rightPicGray = cv2.cvtColor(rightPic, cv2.COLOR_BGR2GRAY)
cv2.imshow("Left panorama gray", leftPicGray)
cv2.imshow("Right panorama gray", rightPicGray)

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=8, disp12MaxDiff=1,
            uniquenessRatio = 10, speckleWindowSize = 10, speckleRange = 8)

disparity = stereo.compute(leftPicGray, rightPicGray).astype(np.float32)
disparity = cv2.normalize(disparity, 0, 255, cv2.NORM_MINMAX)

# mini_disparity = cv2.resize(disparity, (disparity.shape[0] // 6, disparity.shape[1] // 6))

cv2.imshow("Disparity", disparity)
cv2.waitKey(0)
