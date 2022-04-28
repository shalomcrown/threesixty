#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#cython: language_level=3, boundscheck=False
#################################################################
#
# Calibration and testing (WX)
#
#################################################################
import glob
import os
import logging
import logging.handlers
import threading
import tempfile

import cv2
import numpy as np
import sys
import wx, wx.grid
from PIL import Image

scriptPath = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger('stereo_wx')

# ------------------------------------------------------------------------------------------
def setupLogging():
    global logger
    logdir = '/usr/local/lib/airobotics/logs/klvplayer'
    logger = logging.getLogger('stereo_wx')
    logger.setLevel(logging.DEBUG)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    handler = logging.handlers.RotatingFileHandler(os.path.join(logdir, 'stereo_wx.log'),
                                                   backupCount=30, maxBytes=1024 * 1024 * 10)

    formatter = logging.Formatter('[%(levelname)s] %(asctime)s  %(filename)s(%(lineno)d)  %(funcName)s %(message)s')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    outhandler = logging.StreamHandler()
    outhandler.setLevel(logging.DEBUG)
    outhandler.setFormatter(formatter)
    logger.addHandler(outhandler)
    logger.debug("Starting up")

#---------------------------------------------------------------------

class WxStereo(wx.Frame):

    if sys.version_info.minor < 7:
        EVT_NEW_FRAME_ID = wx.NewId()
        EVT_NEW_DATA_ID = wx.NewId()
    else:
        EVT_NEW_FRAME_ID = wx.NewIdRef()
        EVT_NEW_DATA_ID = wx.NewIdRef()

    class FrameEvent(wx.PyEvent):
        def __init__(self, data=None, sliderPos=None, sliderMax=None):
            """Init Result Event."""
            wx.PyEvent.__init__(self)
            self.SetEventType(WxStereo.EVT_NEW_FRAME_ID)
            self.data = data
            self.sliderMax = sliderMax
            self.sliderPos = sliderPos

    class DataEvent(wx.PyEvent):
        def __init__(self, data=None):
            """Init Result Event."""
            wx.PyEvent.__init__(self)
            self.SetEventType(WxStereo.EVT_NEW_DATA_ID)
            self.data = data

    def NEW_FRAME(win, func):
        win.Connect(-1, -1, WxStereo.EVT_NEW_FRAME_ID, func)

    def NEW_DATA(win, func):
        win.Connect(-1, -1, WxStereo.EVT_NEW_DATA_ID, func)

    #---------------------------------------------------------------------

    def setupChessboard(self, height, width):
        self.CHECKERBOARD = (height, width)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        self.objpoints = []

        # Creating vector to store vectors of 2D points for each checkerboard image
        self.imgpointsleft = []
        self.imgpointsRight = []

        # Defining the world coordinates for 3D points
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

    #---------------------------------------------------------------------

    def chooseChessboardSize(self, evt):
        dialog =  wx.Dialog(self)
        dialog.SetSize((250, 200))
        dialog.SetTitle("Change Color Depth")

        def closeDialog(e):
            dialog.Destroy()

        def okDialog(e):
            width = widthSpin.Value
            height = heightSpin.Value
            self.setupChessboard(height, width)
            dialog.Destroy()

        pnl = wx.Panel(dialog)
        vbox = wx.GridSizer(rows=3, cols=2, vgap=2, hgap=2)

        vbox.Add(wx.StaticBox(dialog, label='Width'))
        widthSpin = wx.SpinCtrl(dialog, value=str(self.CHECKERBOARD[0]), min=1, max=120)
        vbox.Add(widthSpin)

        vbox.Add(wx.StaticBox(dialog, label='Height'))
        heightSpin = wx.SpinCtrl(dialog, value=str(self.CHECKERBOARD[1]), min=1, max=120)
        vbox.Add(heightSpin)

        okButton = wx.Button(dialog, label='Ok')
        closeButton = wx.Button(dialog, label='Close')
        vbox.Add(okButton)
        vbox.Add(closeButton)

        okButton.Bind(wx.EVT_BUTTON, okDialog)
        closeButton.Bind(wx.EVT_BUTTON, closeDialog)

        dialog.SetSizer(vbox)
        dialog.ShowModal()
        dialog.Destroy()

    #---------------------------------------------------------------------

    def openDevice(self):
        with wx.FileDialog(self, "Open Dev file", defaultDir="/dev", wildcard="Video files|video*",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return None
            pathname = fileDialog.GetPath()
            return pathname

    def chooseLeftInput(self, evt):
        device = self.openDevice()
        if device != None:
            self.leftDevice = device
            self.startVideoThread()

    def chooseRightInput(self, evt):
        device = self.openDevice()
        if device != None:
            self.rightDevice = device
            self.startVideoThread()

    def setSavedPics(self, evt):
        pass

    def readSavedPics(self, evt):
        with wx.DirDialog(self, "Saved pics folder", defaultPath=os.path.expanduser("~/Videos"),
                           style= wx.DD_DIR_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return None
            self.imageSavepath = fileDialog.GetPath()

        self.savedImageNumber = 0
        self.readSavedPictures = True
        self.startVideoThread()

    #---------------------------------------------------------------------

    def updateVideoFrame(self, bitmap, widget):
        logger.debug("Do update")
        dc = wx.BufferedPaintDC(widget)

        try:
            if bitmap is not None and widget is not None:
                h, w = bitmap.shape[:2]
                wxBitmap = wx.Bitmap.FromBuffer(w, h, bitmap)
                dc.Clear()
                dc.DrawBitmap(wxBitmap, 0, 0)
        except Exception as e:
            logger.exception(f"Exception in pixmap update {e}")

    def updateFrameLeft(self, evt):
        self.updateVideoFrame(self.leftWxImageForDisplay, self.leftInputPanel)
        logger.debug("Updated left")

    def updateFrameRight(self, evt):
        self.updateVideoFrame(self.rightWxImageForDisplay, self.rightInputPanel)
        logger.debug("Updated right")

    def onTakeCalibrationPic(self, evt):
        self.takeCalibrationPicture = True
        logger.debug("Take calibration pic")

    def onCalibrationCalc(self, evt):
        logger.debug("Calc calibration")
        self.calibrationCalc()

    # ------------------------------------------------------------------------------------------

    def __init__(self, filename=None):
        wx.Frame.__init__(self, None, title="KLV Video player")

        self.bitmap = None
        self.videoThread = None
        self.stopVideo = False
        self.leftDevice = None
        self.rightDevice = None
        self.capLeft = None
        self.capRight = None
        self.rightWxImageForDisplay = None
        self.leftWxImageForDisplay = None
        self.takeCalibrationPicture = False
        self.readSavedPictures = False
        self.imageSavepath = tempfile.mkdtemp()
        self.savedImageNumber = 0
        self.objpoints = []
        self.imgpointsLeft = []
        self.imgpointsRight = []

        self.setupChessboard(6, 9)
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        toolbar = self.CreateToolBar()
        takePicTool = toolbar.AddTool(wx.ID_ANY, 'Add Pic', wx.Bitmap('chessboard.png'))
        calcCalibTool = toolbar.AddTool(wx.ID_ANY, 'Calibration calc', wx.Bitmap('scale.png'))
        toolbar.Realize()

        self.Bind(wx.EVT_TOOL, self.onTakeCalibrationPic, takePicTool)

        chooseLeftInputItem = fileMenu.Append(wx.ID_ANY, 'Left input...', 'Choose left input')
        self.Bind(wx.EVT_MENU, self.chooseLeftInput, chooseLeftInputItem)

        chooseRightInputItem = fileMenu.Append(wx.ID_ANY, 'Right input...', 'Choose right input')
        self.Bind(wx.EVT_MENU, self.chooseRightInput, chooseRightInputItem)

        chooseChessboardSizeItem = fileMenu.Append(wx.ID_ANY, 'Set chessboard size')
        self.Bind(wx.EVT_MENU, self.chooseChessboardSize, chooseChessboardSizeItem)

        readSavedPicsItem = fileMenu.Append(wx.ID_ANY, 'Read saved pics...')
        self.Bind(wx.EVT_MENU, self.readSavedPics, readSavedPicsItem)

        setSavedPicsItem = fileMenu.Append(wx.ID_ANY, 'Set saved pics folder...')
        self.Bind(wx.EVT_MENU, self.setSavedPics, setSavedPicsItem)

        fileItem = fileMenu.Append(wx.ID_EXIT, 'Quit', 'Quit application')
        self.Bind(wx.EVT_MENU, self.Close, fileItem)
        menubar.Append(fileMenu, '&File')
        self.SetMenuBar(menubar)

        horizontalSplitter = wx.SplitterWindow(self)
        topVerticalSplitter = wx.SplitterWindow(horizontalSplitter)
        bottomVerticalSplitter = wx.SplitterWindow(horizontalSplitter)

        horizontalSplitter.SplitHorizontally(topVerticalSplitter, bottomVerticalSplitter, sashPosition=480)

        self.leftInputPanel = wx.Panel(topVerticalSplitter)
        self.rightInputPanel = wx.Panel(topVerticalSplitter)

        topVerticalSplitter.SplitVertically(self.leftInputPanel, self.rightInputPanel, sashPosition=640)

        self.leftInputPanel.Bind(wx.EVT_PAINT, self.updateFrameLeft)
        self.rightInputPanel.Bind(wx.EVT_PAINT, self.updateFrameRight)
        # self.displayPanel.Bind(wx.EVT_SIZE, self.videoDisplaySizeChanged)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(horizontalSplitter, proportion=1, flag=wx.EXPAND)
        self.SetSizer(sizer)

        self.SetSize(0, 0, 1280, 960)
        self.Center()
        self.Show()

    #---------------------------------------------------------------------

    def startVideoThread(self):
        if self.videoThread:
            self.stopVideo = True
            self.videoThread.join(5)

        self.videoThread = threading.Thread(target=self.playVideo)
        self.videoThread.setDaemon(True)
        self.videoThread.start()

    #---------------------------------------------------------------------

    def resizeWithAspectRatio(self, inputMat, target):
        height, width = inputMat.shape[:2]
        aspectRatio = width / height
        windowWidth = target.GetSize().x
        windowHeight = target.GetSize().y

        if windowWidth > windowHeight * aspectRatio:
            newImageWidth = int(windowHeight * aspectRatio)
            newImageHeight = int(windowHeight)
        else:
            newImageWidth = int(windowWidth)
            newImageHeight = int(windowWidth / aspectRatio)

        return cv2.resize(inputMat, (newImageWidth, newImageHeight), interpolation=cv2.INTER_LINEAR)


    def displayLeftInputImage(self, inputMat):
        if inputMat is None or not inputMat.size:
            return
        logger.debug("Display left image")
        imageForDisplay = cv2.cvtColor(inputMat.copy(), cv2.COLOR_BGR2RGB)
        self.leftWxImageForDisplay = self.resizeWithAspectRatio(imageForDisplay, self.leftInputPanel)
        self.leftInputPanel.Refresh()


    def displayRightInputImage(self, inputMat):
        if inputMat is None or not inputMat.size:
            return
        logger.debug("Display right image")
        imageForDisplay = cv2.cvtColor(inputMat.copy(), cv2.COLOR_BGR2RGB)
        self.rightWxImageForDisplay = self.resizeWithAspectRatio(imageForDisplay, self.rightInputPanel)
        self.rightInputPanel.Refresh()

    #---------------------------------------------------------------------

    def playVideo(self):
        self.stopVideo = False
        self.paused = False
        leftImage = None
        rightImage = None
        storedPic = False
        rightOK = False
        leftOK = False
        self.savedImageNumber = 0

        if self.capLeft is not None:
            self.capLeft.release()

        if self.capRight is not None:
            self.capRight.release()

        if self.leftDevice is not None:
            self.capLeft = cv2.VideoCapture(self.leftDevice)
            self.capLeft.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capLeft.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if self.rightDevice is not None:
            self.capRight = cv2.VideoCapture(self.rightDevice)
            self.capRight.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capRight.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while not self.stopVideo:
            if self.readSavedPictures:
                leftImageFile = os.path.join(self.imageSavepath, "leftImage_" + str(self.savedImageNumber) + ".jpg")
                rightImageFile = os.path.join(self.imageSavepath, "rightImage_" + str(self.savedImageNumber) + ".jpg")

                if os.path.exists(leftImageFile) and os.path.exists(rightImageFile):
                    leftImage = cv2.imread(leftImageFile)
                    rightImage = cv2.imread(rightImageFile)

                    storedPic = True
                    self.savedImageNumber = self.savedImageNumber + 1
                else:
                    self.readSavedPictures = False
                    continue
            else:
                storedPic = False
                if self.capRight is not None:
                    _, rightImage = self.capRight.read()

                if self.capLeft is not None:
                    _, leftImage = self.capLeft.read()

            if leftImage is None or rightImage is None:
                self.displayLeftInputImage(leftImage)
                self.displayRightInputImage(rightImage)
                continue

            self.grayLeft = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
            self.grayRight = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

            leftRet, cornersLeft = cv2.findChessboardCorners(self.grayLeft, self.CHECKERBOARD,
                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            rightRet, cornersRight = cv2.findChessboardCorners(self.grayRight, self.CHECKERBOARD,
                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if not leftRet or not rightRet:
                self.displayLeftInputImage(leftImage)
                self.displayRightInputImage(rightImage)
                continue

            # print("Left corners", cornersLeft)
            # print("Right corners", cornersRight)

            cornersLeft2 = cv2.cornerSubPix(self.grayLeft, cornersLeft, (11, 11), (-1, -1), self.criteria)
            cornersRight2 = cv2.cornerSubPix(self.grayRight, cornersRight, (11, 11), (-1, -1), self.criteria)

            displayLeft = cv2.drawChessboardCorners(leftImage.copy(), self.CHECKERBOARD, cornersLeft2, leftRet)
            displayRight = cv2.drawChessboardCorners(rightImage.copy(), self.CHECKERBOARD, cornersRight2, rightRet)

            self.displayLeftInputImage(displayLeft)
            self.displayRightInputImage(displayRight)

            # print("Left corners improved", cornersLeft)
            # print("Right corners improved", cornersRight)

            if (self.takeCalibrationPicture or storedPic) and  len(cornersLeft2) and len(cornersRight2):
                self.takeCalibrationPicture = False

                if self.imageSavepath is not None and not storedPic:
                    cv2.imwrite(os.path.join(self.imageSavepath, "leftImage_" + str(self.savedImageNumber) + ".jpg"), leftImage)
                    cv2.imwrite(os.path.join(self.imageSavepath, "rightImage_" + str(self.savedImageNumber) + ".jpg"), rightImage)
                    self.savedImageNumber = self.savedImageNumber + 1;

                self.imgpointsLeft.append(cornersLeft)
                self.imgpointsRight.append(cornersRight)

            else:
                self.takeCalibrationPicture = False

    #---------------------------------------------------------------------

    def calibrationCalc(self):
        retvalLeft, cameraMatrixLeft, distCoeffsLeft, rvecsLeft, tvecsLeft  = cv2.calibrateCamera(self.objp,
                                                                    self.imgpointsLeft,
                                                                    self.grayLeft.shape[::-1],
                                                                    None, None)
        hL, wL = self.grayLeft.shape[:2]

        newCameraMatrixLeft, roiLeft= cv2.getOptimalNewCameraMatrix(cameraMatrixLeft, distCoeffsLeft, (wL,hL),1,(wL,hL))

        retvalRight, cameraMatrixRight, distCoeffsRight, rvecsRight, tvecsRight = cv2.calibrateCamera(self.objp,
                                                                self.imgpointsRight,
                                                                self.greyRight.shape[::-1],
                                                                None,None)
        hR, wR= self.greyRight.shape[:2]
        newCameraMatrixRight, roiRight = cv2.getOptimalNewCameraMatrix(cameraMatrixRight, distCoeffsRight, (wR,hR),1,(wR,hR))

# ------------------------------------------------------------------------------------------

if __name__ == '__main__':
    setupLogging();

    foldername = sys.argv[1] if len(sys.argv) > 1 else None
    app = wx.App()
    WxStereo(foldername)
    app.MainLoop()