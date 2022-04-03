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
import wx, wx.grid

if len(sys.argv) < 2:
    print("Usage: calibration.py  <outputFolder>  [<left input device> <right input device>]")
    print("If no devices are given, existing files will be used")
    print("Too few command line arguments\n")
    sys.exit(2)

os.chdir(sys.argv[1])

# Defining the dimensions of checkerboard (minus one in each direction, h, w)
# CHECKERBOARD = (6, 9)
CHECKERBOARD = (7, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []

# Creating vector to store vectors of 2D points for each checkerboard image
imgpointsleft = []
imgpointsRight = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Interactive mode
if len(sys.argv) > 3:
    capLeft = cv2.VideoCapture(sys.argv[2])
    capRight = cv2.VideoCapture(sys.argv[3])

    capLeft.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capLeft.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capRight.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capRight.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    imageNumber = 0
    while capLeft.isOpened() and capRight.isOpened():
        _, leftImage = capLeft.read()
        _, rightImage = capRight.read()

        if leftImage is None or rightImage is None:
            print("Could not read one of the pictures\n")
            sys.exit(2)

        grayLeft = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

        leftRet, cornersLeft = cv2.findChessboardCorners(grayLeft, CHECKERBOARD,
                            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        rightRet, cornersRight = cv2.findChessboardCorners(grayRight, CHECKERBOARD,
                            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if not leftRet or not rightRet:
            cv2.imshow("Left Input", leftImage)
            cv2.imshow("Right Input", rightImage)
            key = cv2.waitKey(100)
            if key == 27:
                break
            continue

        # refining pixel coordinates for given 2d points.
        cornersLeft2 = cv2.cornerSubPix(grayLeft, cornersLeft, (11, 11), (-1, -1), criteria)
        cornersRight2 = cv2.cornerSubPix(grayRight, cornersRight, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        displayLeft = cv2.drawChessboardCorners(leftImage.copy(), CHECKERBOARD, cornersLeft2, leftRet)
        displayRight = cv2.drawChessboardCorners(rightImage.copy(), CHECKERBOARD, cornersRight2, rightRet)

        cv2.imshow("Left Input", displayLeft)
        cv2.imshow("Right Input", displayRight)
        key = cv2.waitKey(100)
        if key == 13 or key == 10 or key == 141:
            objpoints.append(objp)
            imgpointsleft.append(cornersLeft2)
            imgpointsRight.append(cornersRight2)
            cv2.imwrite(f"ImageLeft-{imageNumber}.jpg", leftImage)
            cv2.imwrite(f"ImageRight-{imageNumber}.jpg", rightImage)
            print(f"Captured images {imageNumber}")
            imageNumber += 1
        elif key == 27:
            break
    capRight.release()
    capLeft.release()

else:
    filesList = glob.glob("*.jpg")
    if not filesList:
        print("No JPG files")
        sys.exit(4)

    for imageNumber in range(len(filesList) // 2):
        leftImage = cv2.imread(f"ImageLeft-{imageNumber}.jpg")
        rightImage = cv2.imread(f"ImageRight-{imageNumber}.jpg")

        if leftImage is None or rightImage is None:
            print("Could not read one of the pictures\n")
            break

        grayLeft = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

        leftRet, cornersLeft = cv2.findChessboardCorners(grayLeft, CHECKERBOARD,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        rightRet, cornersRight = cv2.findChessboardCorners(grayRight, CHECKERBOARD,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if not leftRet or not rightRet:
            continue

        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        cornersLeft2 = cv2.cornerSubPix(grayLeft, cornersLeft, (11, 11), (-1, -1), criteria)
        imgpointsleft.append(cornersLeft2)

        cornersRight2 = cv2.cornerSubPix(grayRight, cornersRight, (11, 11), (-1, -1), criteria)
        imgpointsRight.append(cornersRight2)
        print(f"Read image number {imageNumber}")

cv2.destroyAllWindows()

if not objpoints:
    print("Nothing useful found\n")
    sys.exit(5)

print("Calculate camera matrices")
# Calculate initial calibration matrices
retLeft, cameraMatrixLeft, distCoeffsleft, rvecsLeft, tvecsLeft = cv2.calibrateCamera(
        objpoints, imgpointsleft, grayLeft.shape[::-1], None, None)

retRight, cameraMatrixRight, distCoeffsRight, rvecsRight, tvecsRight = cv2.calibrateCamera(
        objpoints, imgpointsRight, grayRight.shape[::-1], None, None)

print(f"Left camera matrix ret:{retLeft}\n", cameraMatrixLeft)
print(f"Right camera matrix ret:{retRight}\n", cameraMatrixRight)

# Calculate optimal camera matrices
print("Calculate optimal camera matrices")
leftHeight,leftWidth= grayLeft.shape[:2]
newCameraMatrixLeft, roiLeft = cv2.getOptimalNewCameraMatrix(cameraMatrixLeft, distCoeffsleft,
                                                             (leftWidth,leftHeight), 1, (leftWidth,leftHeight))

rightHeight, rightWidth= grayRight.shape[:2]
newCameraMatrixRight, roiRight = cv2.getOptimalNewCameraMatrix(cameraMatrixRight, distCoeffsRight,
                                                               (rightWidth, rightHeight), 1, (rightWidth, rightHeight))

print("Optimal left camera matrix:", newCameraMatrixLeft)
print("Optimal right camera matrix:", newCameraMatrixRight)

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print("Stereo calibration")
# This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
    objpoints, imgpointsleft, imgpointsRight, newCameraMatrixLeft, distCoeffsleft, newCameraMatrixRight, distCoeffsRight,
    grayLeft.shape[::-1], criteria_stereo, flags)

print("Stereo rectification")
# Stereo rectification
rectify_scale = 1
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(
    new_mtxL, distL, new_mtxR, distR, grayLeft.shape[::-1], Rot, Trns, rectify_scale,(0,0))

print("Stereo matrices")
# Calculate mapping matrices
Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                             grayLeft.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                              grayRight.shape[::-1], cv2.CV_16SC2)

print("Saving parameters ......")
cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.release()
cv2.destroyAllWindows()
print("Done")
