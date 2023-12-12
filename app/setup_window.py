from PyQt5 import QtWidgets
import sys

# layout
from layouts_window import layout_home_window
from layouts_window import layout_image_analysis_window

# button handler
from button_handlers import button_handler_home_window


class HomeWindow(QtWidgets.QMainWindow, layout_home_window.Ui_home_window):
    """
    Manage the home window.

    Args:
        QtWidgets.QMainWindow (class): Qt Library
        layout_home_window.Ui_home_window (class): layout for homewindow
    """
    def __init__(self):
        super(HomeWindow, self).__init__()
        self.setupUi(self)
        self.connect_handler_to_main_window_button()

    def connect_handler_to_main_window_button(self):
        """ボタンと処置関数を接続"""
        self.pushButton_for_FeatureAnalysis.clicked.connect(lambda: button_handler_home_window.open_window_image_analysis(self))


class ImageAnalysisWindow(QtWidgets.QMainWindow, layout_image_analysis_window.Ui_MainWindow_for_ImageAnalysis):
    """
    Manage the image analysis window.

    Args:
        QtWidgets.QMainWindow (class): Qt Library
        layout_home_window.Ui_home_window (class): layout for homewindow
    """
    def __init__(self):
        super(ImageAnalysisWindow, self).__init__()
        self.setupUi(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    hw = HomeWindow()
    hw.show()
    sys.exit(app.exec_())
