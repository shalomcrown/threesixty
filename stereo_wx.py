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

    def updateOutputFrameLeft(self, evt):
        self.updateVideoFrame(self.leftWxOutputForDisplay, self.leftOutputPanel)
        logger.debug("Updated left")

    def updateOutputFrameRight(self, evt):
        self.updateVideoFrame(self.rightWxOutputForDisplay, self.rightOutputPanel)
        logger.debug("Updated right")


    def onTakeCalibrationPic(self, evt):
        self.takeCalibrationPicture = True
        logger.debug("Take calibration pic")

    def onCalibrationCalc(self, evt):
        logger.debug("Calc calibration")
        self.calibrationCalc()

    def saveCoefficients(self, evt):
        if self.leftStereoMap is None or self.rightStereoMap is None:
            wx.MessageDialog(self, "No calibration coefficients available")
            return

        with wx.FileDialog(self, "Open coefficients file", defaultDir=os.path.expanduser("~/Videos"), wildcard="XML files|*xml",
                           style=wx.FD_SAVE) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            file = cv2.FileStorage(fileDialog.GetPath(), cv2.FILE_STORAGE_WRITE)

            file.write("left_camera_matrix", self.newCameraMatrixLeft)
            file.write("left_distortion_coefficients",  self.distCoeffsLeft)
            file.write("left_stereo_rectification", self.leftRectification)
            file.write("left_stereo_projection", self.projectionMatrixLeft)

            file.write("right_camera_matrix", self.newCameraMatrixRight)
            file.write("right_distortion_coefficients",  self.distCoeffsRight)
            file.write("right_stereo_rectification", self.rightRectification)
            file.write("right_stereo_projection", self.projectionMatrixRight)

            file.write("Left_Stereo_Map_x", self.leftStereoMap[0])
            file.write("Left_Stereo_Map_y", self.leftStereoMap[1])
            file.write("Right_Stereo_Map_x", self.rightStereoMap[0])
            file.write("Right_Stereo_Map_y", self.rightStereoMap[1])
            file.release()

    #---------------------------------------------------------------------

    def loadCoefficients(self, evt):
        with wx.FileDialog(self, "Open coefficients file", defaultDir=os.path.expanduser("~/Videos"), wildcard="XML files|*xml",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            file = cv2.FileStorage(fileDialog.GetPath(), cv2.FILE_STORAGE_READ)
            file.getNode("")

            self.newCameraMatrixLeft = file.getNode("left_camera_matrix").mat()
            self.distCoeffsLeft = file.getNode("left_distortion_coefficients").mat()
            self.leftRectification = file.getNode("left_stereo_rectification").mat()
            self.projectionMatrixLeft = file.getNode("left_stereo_projection").mat()

            self.newCameraMatrixRight = file.getNode("right_camera_matrix").mat()
            self.distCoeffsRight = file.getNode("right_distortion_coefficients").mat()
            self.rightRectification = file.getNode("right_stereo_rectification").mat()
            self.projectionMatrixRight = file.getNode("right_stereo_projection").mat()

            self.leftStereoMap = []
            self.rightStereoMap = []

            self.leftStereoMap.append(file.getNode("Left_Stereo_Map_x").mat())
            self.leftStereoMap.append(file.getNode("Left_Stereo_Map_y").mat())
            self.rightStereoMap.append(file.getNode("Right_Stereo_Map_x").mat())
            self.rightStereoMap.append(file.getNode("Right_Stereo_Map_y").mat())
            file.release()

    # ------------------------------------------------------------------------------------------

    def __init__(self, filename=None):
        wx.Frame.__init__(self, None, title="Stereo test")

        self.bitmap = None
        self.videoThread = None
        self.stopVideo = False
        self.leftDevice = None
        self.rightDevice = None
        self.capLeft = None
        self.capRight = None
        self.rightWxImageForDisplay = None
        self.leftWxImageForDisplay = None
        self.rightWxOutputForDisplay = None
        self.leftWxOutputForDisplay = None
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
        self.Bind(wx.EVT_TOOL, self.onCalibrationCalc, calcCalibTool)

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

        saveCoefficientsItem = fileMenu.Append(wx.ID_ANY, 'Save calibration...')
        self.Bind(wx.EVT_MENU, self.saveCoefficients, saveCoefficientsItem)

        loadCoefficientsItem = fileMenu.Append(wx.ID_ANY, 'Load calibration...')
        self.Bind(wx.EVT_MENU, self.loadCoefficients, loadCoefficientsItem)

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

        self.leftOutputPanel = wx.Panel(bottomVerticalSplitter)
        self.rightOutputPanel = wx.Panel(bottomVerticalSplitter)

        bottomVerticalSplitter.SplitVertically(self.leftOutputPanel, self.rightOutputPanel, sashPosition=640)

        self.leftOutputPanel.Bind(wx.EVT_PAINT, self.updateOutputFrameLeft)
        self.rightOutputPanel.Bind(wx.EVT_PAINT, self.updateOutputFrameRight)

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

    def displayLeftOutputImage(self, inputMat):
        if inputMat is None or not inputMat.size:
            return
        logger.debug("Display left image")
        imageForDisplay = cv2.cvtColor(inputMat.copy(), cv2.COLOR_BGR2RGB)
        self.leftWxOutputForDisplay = self.resizeWithAspectRatio(imageForDisplay, self.leftOutputPanel)
        self.leftOutputPanel.Refresh()


    def displayRightOuputImage(self, inputMat):
        if inputMat is None or not inputMat.size:
            return
        logger.debug("Display right image")
        imageForDisplay = cv2.cvtColor(inputMat.copy(), cv2.COLOR_BGR2RGB)
        self.rightWxOutputForDisplay = self.resizeWithAspectRatio(imageForDisplay, self.rightOutputPanel)
        self.rightOutputPanel.Refresh()

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
        self.cameraMatrixLeft = None
        self.distCoeffsLeft = None
        self.rvecsLeft = None
        self.tvecsLeft = None
        self.newCameraMatrixLeft = None
        self.roiLeft = None
        self.cameraMatrixRight = None
        self.distCoeffsRight = None
        self.rvecsRight = None
        self.tvecsRight = None
        self.newCameraMatrixRight = None
        self.roiRight = None
        self.rotation = self.translation = self.essential = self.fundamental = None
        self.leftRectification = self.rightRectification = self.projectionMatrixLeft = self.projectionMatrixRight = None
        self.Qmatrix = self.leftROI = self.rightROI = None
        self.leftStereoMap = self.rightStereoMap = None

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
                    leftImage = rightImage = None
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

            if leftRet and rightRet:
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
                    self.objpoints.append(self.objp)

                else:
                    self.takeCalibrationPicture = False
            else:
                self.displayLeftInputImage(leftImage)
                self.displayRightInputImage(rightImage)
                self.takeCalibrationPicture = False


            if self.leftStereoMap is not None and len(self.leftStereoMap) and \
                            self.rightStereoMap is not None and len(self.rightStereoMap):
                
                leftRectifiedImage = cv2.remap(leftImage, self.leftStereoMap[0], self.leftStereoMap[1], 
                                               cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                
                rightRectifiedImage = cv2.remap(rightImage, self.rightStereoMap[0], self.rightStereoMap[1], 
                                               cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

                outputAnaglyph = leftRectifiedImage.copy()
                outputAnaglyph[:, :, 0] = leftRectifiedImage[:, :, 0]
                outputAnaglyph[:, :, 1] = leftRectifiedImage[:, :, 1]
                outputAnaglyph[:, :, 2] = rightRectifiedImage[:, :, 2]

                self.displayRightOuputImage(outputAnaglyph)

    #---------------------------------------------------------------------

    def calibrationCalc(self):
        retvalLeft, self.cameraMatrixLeft, self.distCoeffsLeft, self.rvecsLeft, self.tvecsLeft  = cv2.calibrateCamera(
                                                                    self.objpoints,
                                                                    self.imgpointsLeft,
                                                                    self.grayLeft.shape[::-1],
                                                                    None, None)
        hL, wL = self.grayLeft.shape[:2]

        self.newCameraMatrixLeft, self.roiLeft  = cv2.getOptimalNewCameraMatrix(self.cameraMatrixLeft, self.distCoeffsLeft, (wL,hL),1,(wL,hL))

        retvalRight, self.cameraMatrixRight, self.distCoeffsRight, self.rvecsRight, self.tvecsRight = cv2.calibrateCamera(
                                                                self.objpoints,
                                                                self.imgpointsRight,
                                                                self.grayRight.shape[::-1],
                                                                None,None)
        hR, wR= self.grayRight.shape[:2]
        self.newCameraMatrixRight, self.roiRight = cv2.getOptimalNewCameraMatrix(self.cameraMatrixRight, self.distCoeffsRight, (wR,hR),1,(wR,hR))

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        retStereo, self.newCameraMatrixLeft, self.distCoeffsLeft, self.newCameraMatrixRight, self.distCoeffsRight, \
        self.rotation, self.translation, self.essential, self.fundamental = cv2.stereoCalibrate(
                            self.objpoints,
                            self.imgpointsLeft,
                            self.imgpointsRight,
                            self.newCameraMatrixLeft,
                            self.distCoeffsLeft,
                            self.newCameraMatrixRight,
                            self.distCoeffsRight,
                            self.grayLeft.shape[::-1],
                            criteria_stereo, flags)

        rectify_scale = 1
        self.leftRectification, self.rightRectification, self.projectionMatrixLeft, self.projectionMatrixRight, \
        self.Qmatrix, self.leftROI, self.rightROI = cv2.stereoRectify(
                            self.newCameraMatrixLeft,  self.distCoeffsLeft,  self.newCameraMatrixRight,  self.distCoeffsRight,
                            self.grayLeft.shape[::-1],
                            self.rotation, self.translation,
                            rectify_scale, (0, 0))

        self.leftStereoMap =  cv2.initUndistortRectifyMap(
                                    self.newCameraMatrixLeft,
                                    self.distCoeffsLeft,
                                    self.leftRectification, self.projectionMatrixLeft,
                                    self.grayLeft.shape[::-1], cv2.CV_16SC2)

        self.rightStereoMap =  cv2.initUndistortRectifyMap(
                                    self.newCameraMatrixRight,
                                    self.distCoeffsRight,
                                    self.rightRectification, self.projectionMatrixRight,
                                    self.grayRight.shape[::-1], cv2.CV_16SC2)

# ------------------------------------------------------------------------------------------

if __name__ == '__main__':
    setupLogging();

    foldername = sys.argv[1] if len(sys.argv) > 1 else None
    app = wx.App()
    WxStereo(foldername)
    app.MainLoop()