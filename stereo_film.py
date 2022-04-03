#!/usr/bin/env python
#######################################################################
#
# Calibrate a pair of stereo cameras. It is assumed that they
# are fixed in place relative to each other, and as parallel as possible.
#
#######################################################################
import glob
import os

import cv2
import numpy as np
import sys


if len(sys.argv) < 3:
    print("Usage: stereo_film.py  <calibrationFile>  <left input device> <right input device>")
    print("If no devices are given, existing files will be used")
    print("Too few command line arguments\n")
    sys.exit(2)

def nothing(x):
    pass

def mouseCallback(event, x, y, flags, userdata):
    disparity = disp[x, y]
    print(f"Disparity {disparity} ({x},{y})")


# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage(sys.argv[1], cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

# Setting parameters for StereoSGBM algorithm
minDisparity = 0;
numDisparities = 64;
blockSize = 8;
disp12MaxDiff = 1;
uniquenessRatio = 10;
speckleWindowSize = 10;
speckleRange = 8;

# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )

# stereo = cv2.StereoBM_create()
capLeft = cv2.VideoCapture(sys.argv[2])
capRight = cv2.VideoCapture(sys.argv[3])

rectifiedDispWindowName = "Rectified Disparity map"

cv2.namedWindow(rectifiedDispWindowName)
cv2.resizeWindow(rectifiedDispWindowName, 600, 600)

cv2.createTrackbar('numDisparities', rectifiedDispWindowName,1,17,nothing)
cv2.createTrackbar('blockSize', rectifiedDispWindowName,5,50,nothing)
cv2.createTrackbar('preFilterType', rectifiedDispWindowName,1,1,nothing)
cv2.createTrackbar('preFilterSize',rectifiedDispWindowName,2,25,nothing)
cv2.createTrackbar('preFilterCap',rectifiedDispWindowName,5,62,nothing)
cv2.createTrackbar('textureThreshold',rectifiedDispWindowName,10,100,nothing)
cv2.createTrackbar('uniquenessRatio',rectifiedDispWindowName,15,100,nothing)
cv2.createTrackbar('speckleRange',rectifiedDispWindowName,0,100,nothing)
cv2.createTrackbar('speckleWindowSize',rectifiedDispWindowName,3,25,nothing)
cv2.createTrackbar('disp12MaxDiff',rectifiedDispWindowName,5,25,nothing)
cv2.createTrackbar('minDisparity',rectifiedDispWindowName,5,25,nothing)
cv2.setMouseCallback(rectifiedDispWindowName, mouseCallback)


while capLeft.isOpened() and capRight.isOpened():
    _, leftImage = capLeft.read()
    _, rightImage = capRight.read()

    if leftImage is None or rightImage is None:
        print("Could not read one of the pictures\n")
        sys.exit(2)

    cv2.imshow("Left image before rectification", leftImage)
    cv2.imshow("Right image before rectification", rightImage)

    leftRectified = cv2.remap(leftImage, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    rightRectified = cv2.remap(rightImage, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    cv2.imshow("Left image after rectification", leftRectified)
    cv2.imshow("Right image after rectification", rightRectified)

    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities', rectifiedDispWindowName) * 16
    blockSize = cv2.getTrackbarPos('blockSize', rectifiedDispWindowName) * 2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType', rectifiedDispWindowName)
    preFilterSize = cv2.getTrackbarPos('preFilterSize', rectifiedDispWindowName) * 2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap', rectifiedDispWindowName)
    textureThreshold = cv2.getTrackbarPos('textureThreshold', rectifiedDispWindowName)
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', rectifiedDispWindowName)
    speckleRange = cv2.getTrackbarPos('speckleRange', rectifiedDispWindowName)
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', rectifiedDispWindowName) * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', rectifiedDispWindowName)
    minDisparity = cv2.getTrackbarPos('minDisparity', rectifiedDispWindowName)

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    # stereo.setPreFilterType(preFilterType)
    # stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    # stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    # out = rightRectified.copy()
    # out[:,:,0] = rightRectified[:,:,0]
    # out[:,:,1] = rightRectified[:,:,1]
    # out[:,:,2] = leftRectified[:,:,2]

    out = leftRectified.copy()
    out[:,:,0] = leftRectified[:,:,0]
    out[:,:,1] = leftRectified[:,:,1]
    out[:,:,2] = rightRectified[:,:,2]

    cv2.imshow("Rectified Output image", out)

    grayLeftRectified = cv2.cvtColor(leftRectified, cv2.COLOR_BGR2GRAY)
    grayRightRectified = cv2.cvtColor(rightRectified, cv2.COLOR_BGR2GRAY)

    disp = stereo.compute(grayLeftRectified, grayRightRectified).astype(np.float32)
    dispNormalilzed = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow(rectifiedDispWindowName, dispNormalilzed)

    out = leftImage.copy()
    out[:,:,0] = leftImage[:,:,0]
    out[:,:,1] = leftImage[:,:,1]
    out[:,:,2] = rightImage[:,:,2]
    cv2.imshow("Unrectified Output image", out)

    grayLeft = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

    dispUnrectified = stereo.compute(grayLeft, grayRight).astype(np.float32)
    dispUnrectified = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("Unrectified Disparity map", dispUnrectified)

    key = cv2.waitKey(20)

    if key == 27:
        break

capLeft.release()
capRight.release()
cv2.destroyAllWindows()