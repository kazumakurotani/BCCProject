from PyQt5 import QtCore, QtWidgets
import sys

# layout
from layout import layout_home_window

# ボタン操作に応じた処理関数


class MainWindow(QtWidgets.QMainWindow, layout_home_window.Ui_home_window):
    """_summary_

    Args:
        QtWidgets.QMainWindow (class): Qt Library
        layout_home_window.Ui_home_window (class): layout for homewindow
    """
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.connect_handler_to_main_window_button()

    def connect_handler_to_main_window_button(self):
        """ボタンと処置関数を接続"""
        self.button_openImageAnalysisWindow.clicked.connect(lambda: button_handler_main_window.openWindow_imageAnalysis(self))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
