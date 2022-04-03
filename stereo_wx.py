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

import cv2
import numpy as np
import sys
import wx, wx.grid
from PIL import Image


# ------------------------------------------------------------------------------------------
def setupLogging():
    logdir = '/usr/local/lib/airobotics/logs/klvplayer'
    logger = logging.getLogger('sereo_wx')
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

        self.setupChessboard(6, 9)
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()

        chooseLeftInputItem = fileMenu.Append(wx.ID_ANY, 'Left input...', 'Choose left input')
        self.Bind(wx.EVT_MENU, self.chooseLeftInput, chooseLeftInputItem)

        chooseRightInputItem = fileMenu.Append(wx.ID_ANY, 'Right input...', 'Choose right input')
        self.Bind(wx.EVT_MENU, self.chooseRightInput, chooseRightInputItem)

        chooseChessboardSizeItem = fileMenu.Append(wx.ID_ANY, 'Set chessboard size')
        self.Bind(wx.EVT_MENU, self.chooseChessboardSize, chooseChessboardSizeItem)

        fileItem = fileMenu.Append(wx.ID_EXIT, 'Quit', 'Quit application')
        self.Bind(wx.EVT_MENU, self.Close, fileItem)
        menubar.Append(fileMenu, '&File')
        self.SetMenuBar(menubar)

        horizontalSplitter = wx.SplitterWindow(self)
        topVerticalSplitter = wx.SplitterWindow(horizontalSplitter)
        bottomVerticalSplitter = wx.SplitterWindow(horizontalSplitter)

        horizontalSplitter.SplitHorizontally(topVerticalSplitter, bottomVerticalSplitter, sashPosition=480)

        self.leftInputPanel = wx.Panel(topVerticalSplitter)
        # self.displayPanel.Bind(wx.EVT_PAINT, self.updateFrame)
        # self.displayPanel.Bind(wx.EVT_SIZE, self.videoDisplaySizeChanged)

        self.rightInputPanel = wx.Panel(topVerticalSplitter)

        topVerticalSplitter.SplitVertically(self.leftInputPanel, self.rightInputPanel, sashPosition=640)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(horizontalSplitter, proportion=1, flag=wx.EXPAND)
        self.SetSizer(sizer)

        self.SetSize(0, 0, 1280 + 200, 960)
        # self.videoPanelSize = self.displayPanel.GetSize()
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

    def playVideo(self):
        self.stopVideo = False
        self.paused = False

        if self.capLeft is not None:
            self.capLeft.release()

        if self.capRight is not None:
            self.capRight.release()

        if self.leftDevice is not None:
            self.capLeft = cv2.VideoCapture(self.leftDevice)
            self.capLeft.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capLeft.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if self.RightDevice is not None:
            self.capRight = cv2.VideoCapture(self.RightDevice)
            self.capRight.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capRight.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while not self.stopVideo:
            if self.capRight is not None:
                _, rightImage = self.capRight.read()
                image = rightImage.to_image().copy()

            if self.capLeft is not None:
                _, leftImage = self.capLeft.read()

        wx.PostEvent(self, WxStereo.DataEvent(None))




#---------------------------------------------------------------------

def exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

# ------------------------------------------------------------------------------------------

if __name__ == '__main__':
    setupLogging();

    sys._excepthook = sys.excepthook
    sys.excepthook = exception_hook

    foldername = sys.argv[1] if len(sys.argv) > 1 else None
    app = wx.App()
    WxStereo(foldername)
    app.MainLoop()