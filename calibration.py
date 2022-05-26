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
    logger = logging.getLogger('camera_calibration')
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

class Calibrate(wx.Frame):

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
            self.SetEventType(Calibrate.EVT_NEW_FRAME_ID)
            self.data = data
            self.sliderMax = sliderMax
            self.sliderPos = sliderPos

    class DataEvent(wx.PyEvent):
        def __init__(self, data=None):
            """Init Result Event."""
            wx.PyEvent.__init__(self)
            self.SetEventType(Calibrate.EVT_NEW_DATA_ID)
            self.data = data

    def NEW_FRAME(win, func):
        win.Connect(-1, -1, Calibrate.EVT_NEW_FRAME_ID, func)

    def NEW_DATA(win, func):
        win.Connect(-1, -1, Calibrate.EVT_NEW_DATA_ID, func)

    #---------------------------------------------------------------------

    def setupChessboard(self, height, width):
        self.CHECKERBOARD = (height, width)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        self.objpoints = []

        # Creating vector to store vectors of 2D points for each checkerboard image
        self.imgpointsleft = []

        # Defining the world coordinates for 3D points
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

    #---------------------------------------------------------------------

    def chooseChessboardSize(self, evt):
        dialog =  wx.Dialog(self)
        dialog.SetSize((250, 200))
        dialog.SetTitle("Set chessboard size")

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

    def onFocalLength(self, evt):
        dialog = wx.Dialog(self)
        dialog.SetSize((400, 200))
        dialog.SetTitle("Set chessboard distance")

        def closeDialog(e):
            dialog.Destroy()

        def okDialog(e):
            self.chessboardDistance = float(distanceSpin.Value)
            self.chessboardDistancePhysicalWidth = float(widthhSpin.Value)

            dialog.Destroy()

        pnl = wx.Panel(dialog)
        vbox = wx.GridSizer(rows=3, cols=2, vgap=2, hgap=2)

        vbox.Add(wx.StaticBox(dialog, label='Distance from camera (meters)'))
        distanceSpin = wx.TextCtrl(dialog, value=str(self.chessboardDistance))
        vbox.Add(distanceSpin)

        vbox.Add(wx.StaticBox(dialog, label='Width of chessboard inner (meters)'))
        widthhSpin = wx.TextCtrl(dialog, value=str(self.chessboardDistancePhysicalWidth))
        vbox.Add(widthhSpin)

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

    def chooseInput(self, evt):
        device = self.openDevice()
        if device != None:
            self.leftDevice = device
            self.startVideoThread()

    def setSavedPics(self, evt):
        with wx.DirDialog(self, "Saved pics folder", defaultPath=os.path.expanduser("~/Videos"),
                           style= wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return None
            self.imageSavepath = fileDialog.GetPath()

    def readSavedPics(self, evt):
        with wx.DirDialog(self, "Saved pics folder", defaultPath=os.path.expanduser("~/Videos"),
                           style= wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as fileDialog:
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
                if not h or not w:
                    dc.Clear()
                    return

                wxBitmap = wx.Bitmap.FromBuffer(w, h, bitmap)
                dc.Clear()
                dc.DrawBitmap(wxBitmap, 0, 0)
        except Exception as e:
            logger.exception(f"Exception in pixmap update {e}")

    def updateFrameLeft(self, evt):
        self.updateVideoFrame(self.leftWxImageForDisplay, self.leftInputPanel)
        logger.debug("Updated left")

    def updateOutputFrameLeft(self, evt):
        self.updateVideoFrame(self.leftWxOutputForDisplay, self.leftOutputPanel)
        logger.debug("Updated left")

    #---------------------------------------------------------------------

    def onTakeCalibrationPic(self, evt):
        self.takeCalibrationPicture = True
        logger.debug("Take calibration pic")

    def onCalibrationCalc(self, evt):
        logger.debug("Calc calibration")
        self.calibrationCalc()

    #---------------------------------------------------------------------

    def saveCoefficients(self, evt):
        if self.newCameraMatrixLeft is None:
            dlg = wx.MessageDialog(self, "No calibration coefficients available")
            dlg.ShowModal()
            return

        with wx.FileDialog(self, "Open coefficients file", defaultDir=os.path.expanduser("~/Videos"), wildcard="XML files|*xml",
                           style=wx.FD_SAVE) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            file = cv2.FileStorage(fileDialog.GetPath(), cv2.FILE_STORAGE_WRITE)

            file.write("camera_matrix", self.newCameraMatrixLeft)
            file.write("distortion_coefficients",  self.distCoeffsLeft)
            file.write("focal_length", self.focalLength)
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
            file.release()

    # ------------------------------------------------------------------------------------------

    def __init__(self, filename=None):
        wx.Frame.__init__(self, None, title="Single camera calibration")

        self.bitmap = None
        self.videoThread = None
        self.stopVideo = False
        self.leftDevice = None
        self.capLeft = None
        self.leftWxImageForDisplay = None
        self.leftWxOutputForDisplay = None
        self.takeCalibrationPicture = False
        self.readSavedPictures = False
        self.imageSavepath = tempfile.mkdtemp()
        self.savedImageNumber = 0
        self.objpoints = []
        self.imgpointsLeft = []
        self.stereo = cv2.StereoBM_create()
        self.newCameraMatrixLeft = None
        self.chessboardDistancePhysicalWidth = 0.0335;
        self.chessboardDistance = 2.010;

        self.setupChessboard(6, 9)
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        toolbar = self.CreateToolBar()
        takePicTool = toolbar.AddTool(wx.ID_ANY, 'Add Pic', wx.Bitmap('chessboard.png'))
        calcCalibTool = toolbar.AddTool(wx.ID_ANY, 'Calibration calc', wx.Bitmap('calculator.png'))
        focalLengthTool = toolbar.AddTool(wx.ID_ANY, 'Do focal length calculation calc', wx.Bitmap('scale.png'))

        toolbar.Realize()

        self.Bind(wx.EVT_TOOL, self.onTakeCalibrationPic, takePicTool)
        self.Bind(wx.EVT_TOOL, self.onCalibrationCalc, calcCalibTool)
        self.Bind(wx.EVT_TOOL, self.onFocalLength, focalLengthTool)

        chooseLeftInputItem = fileMenu.Append(wx.ID_ANY, 'Input...', 'Choose input device')
        self.Bind(wx.EVT_MENU, self.chooseInput, chooseLeftInputItem)

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

        self.leftInputPanel = wx.Panel(horizontalSplitter)
        self.leftInputPanel.Bind(wx.EVT_PAINT, self.updateFrameLeft)
        self.leftOutputPanel = wx.Panel(horizontalSplitter)
        self.leftOutputPanel.Bind(wx.EVT_PAINT, self.updateOutputFrameLeft)

        horizontalSplitter.SplitHorizontally(self.leftInputPanel, self.leftOutputPanel, sashPosition=480)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(horizontalSplitter, proportion=3, flag=wx.EXPAND)
        sizer.AddSpacer(5)

        self.SetSizer(sizer)

        self.SetSize(0, 0, 1280 * 3 // 2, 960)
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


    def displayInputImage(self, inputMat):
        if inputMat is None or not inputMat.size:
            return
        logger.debug("Display left image")
        imageForDisplay = cv2.cvtColor(inputMat.copy(), cv2.COLOR_BGR2RGB)
        self.leftWxImageForDisplay = self.resizeWithAspectRatio(imageForDisplay, self.leftInputPanel)
        self.leftInputPanel.Refresh()


    def displayOutputImage(self, inputMat):
        if inputMat is None or not inputMat.size:
            return
        logger.debug("Display left image")
        imageForDisplay = cv2.cvtColor(inputMat.copy(), cv2.COLOR_BGR2RGB)
        self.leftWxOutputForDisplay = self.resizeWithAspectRatio(imageForDisplay, self.leftOutputPanel)
        self.leftOutputPanel.Refresh()

    #---------------------------------------------------------------------

    def playVideo(self):
        self.stopVideo = False
        self.paused = False
        leftImage = None
        storedPic = False
        leftOK = False
        self.savedImageNumber = 0
        self.cameraMatrixLeft = None
        self.distCoeffsLeft = None
        self.rvecsLeft = None
        self.tvecsLeft = None
        self.roiLeft = None
        self.rotation = self.translation = self.essential = self.fundamental = None
        self.leftRectification = self.projectionMatrixLeft = None
        self.Qmatrix = self.leftROI = None
        self.leftStereoMap = None

        if self.capLeft is not None:
            self.capLeft.release()

        if self.leftDevice is not None:
            self.capLeft = cv2.VideoCapture(self.leftDevice)
            self.capLeft.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capLeft.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while not self.stopVideo:
            if self.readSavedPictures:
                leftImageFile = os.path.join(self.imageSavepath, "image_" + str(self.savedImageNumber) + ".jpg")

                if os.path.exists(leftImageFile):
                    leftImage = cv2.imread(leftImageFile)
                    storedPic = True
                    self.savedImageNumber = self.savedImageNumber + 1
                else:
                    self.readSavedPictures = False
                    leftImage = None
                    continue
            else:
                storedPic = False
                if self.capLeft is not None:
                    _, leftImage = self.capLeft.read()

            if leftImage is None:
                self.displayInputImage(leftImage)
                continue

            self.grayLeft = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)

            leftRet, cornersLeft = cv2.findChessboardCorners(self.grayLeft, self.CHECKERBOARD,
                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if leftRet:
                cornersLeft2 = cv2.cornerSubPix(self.grayLeft, cornersLeft, (11, 11), (-1, -1), self.criteria)
                displayLeft = cv2.drawChessboardCorners(leftImage.copy(), self.CHECKERBOARD, cornersLeft2, leftRet)
                self.displayInputImage(displayLeft)

                sortedByX = sorted(cornersLeft2[:,0], key=lambda x: x[0]);
                innerWidth = sortedByX[-1][0] - sortedByX[0][0]

                sortedByY = sorted(cornersLeft2[:,0], key=lambda x: x[1]);
                innerHeight = sortedByY[-1][1] - sortedByY[0][1]

                self.focalLength = (innerWidth * self.chessboardDistance) / self.chessboardDistancePhysicalWidth

                if (self.takeCalibrationPicture or storedPic) and  len(cornersLeft2):
                    self.takeCalibrationPicture = False

                    if self.imageSavepath is not None and not storedPic:
                        cv2.imwrite(os.path.join(self.imageSavepath, "image_" + str(self.savedImageNumber) + ".jpg"), leftImage)
                        self.savedImageNumber = self.savedImageNumber + 1;

                    self.imgpointsLeft.append(cornersLeft)
                    self.objpoints.append(self.objp)

                else:
                    self.takeCalibrationPicture = False
            else:
                self.displayInputImage(leftImage)
                self.takeCalibrationPicture = False

    #---------------------------------------------------------------------

    def calibrationCalc(self):
        if not self.savedImageNumber:
            logger.debug("No images for calibration")
            return

        retvalLeft, self.cameraMatrixLeft, self.distCoeffsLeft, self.rvecsLeft, self.tvecsLeft  = cv2.calibrateCamera(
                                                                    self.objpoints,
                                                                    self.imgpointsLeft,
                                                                    self.grayLeft.shape[::-1],
                                                                    None, None)
        hL, wL = self.grayLeft.shape[:2]
        self.newCameraMatrixLeft, self.roiLeft  = cv2.getOptimalNewCameraMatrix(self.cameraMatrixLeft, self.distCoeffsLeft, (wL,hL),1,(wL,hL))


# ------------------------------------------------------------------------------------------

if __name__ == '__main__':
    setupLogging();

    foldername = sys.argv[1] if len(sys.argv) > 1 else None
    app = wx.App()
    Calibrate(foldername)
    app.MainLoop()