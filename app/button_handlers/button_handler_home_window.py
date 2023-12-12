from PyQt5 import QtCore, QtGui, QtWidgets
import setup_window
import sys

def open_window_image_analysis(HomeWindow):
    # ウィンドウを開く
    setup_window.imageAnalysisWindow = setup_window.ImageAnalysisWindow()
    setup_window.imageAnalysisWindow.show()
