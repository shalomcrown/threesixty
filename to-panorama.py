#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#cython: language_level=3, boundscheck=False
###############################################################################
#
# Script to convert Kodak360P 360 degree video to panoramic
#
###############################################################################
import cv2 as cv
import sys


if len(sys.argv) < 3:
    print("Usage: python to-panorama.py <input file> <output file>\n")
    print("Too few command line arguments\n")
    sys.exit(2)

cap = cv.VideoCapture(sys.argv[1])
outFile = None

while cap.isOpened():
    success,input_image = cap.read()

    if input_image is None:
        break

    image_size = input_image.shape
    maxRadius = max(image_size[0] / 2.0, image_size[1] / 2.0)

    output_image = cv.warpPolar(input_image, (-1, -1),  (image_size[0] / 2.0, image_size[1] / 2.0),
                                maxRadius, cv.WARP_POLAR_LINEAR)

    output_image = cv.rotate(output_image, cv.ROTATE_90_CLOCKWISE)

    if outFile is None:
        fps = cap.get(cv.CAP_PROP_FPS)
        outFile = cv.VideoWriter(sys.argv[2], cv.VideoWriter_fourcc(*'MP4V'), fps, output_image.shape[1::-1])
    outFile.write(output_image)

if outFile is not None:
    outFile.release()