from PyQt5 import QtCore, QtGui, QtWidgets
import setup_window
import sys

def openWindow_imageAnalysis(HomeWindow):

    # ウィンドウを開く
    setup_window.imageAnalysisWindow = setup_window.ImageAnalysisWindow()
    setup_window.imageAnalysisWindow.show()
