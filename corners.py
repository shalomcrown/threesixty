#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#cython: language_level=3, boundscheck=False
###############################################################################
#
# Playing with stiching 2 360 degree pictures - something
#
###############################################################################

import cv2
import numpy as np
import sys


def harris(grayImage):
    dst = cv2.cornerHarris(grayImage, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(grayImage, np.float32(centroids), (5, 5), (-1, -1), criteria)

    res = np.hstack((centroids, corners))
    res = np.int0(res)

    # Threshold for an optimal value, it may vary depending on the image.
    # image[dst > 0.01 * dst.max()] = [0,0,255]
    image[res[:, 1], res[:, 0]] = [0, 0, 255]
    image[res[:, 3], res[:, 2]] = [0, 255, 0]
    return image

#---------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python corners.py <input file>\n")
    print("Too few command line arguments\n")
    sys.exit(2)

image = cv2.imread(sys.argv[1])

grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayImage = np.float32(grayImage)

mask = np.ones(grayImage.shape, cv2.CV_8UC1)
cv2.rectangle(mask, (0, 0), (200, 200), 0)
cv2.imshow(mask)
cv2.waitKey(0)

corners = cv2.goodFeaturesToTrack(grayImage, 25, 0.01, 10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(image, (x,y), 3, (0, 0, 255), -1)

cv2.imshow('Corners', image)
cv2.waitKey(0)
