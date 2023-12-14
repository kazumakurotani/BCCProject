from PyQt5 import QtWidgets, QtCore
import sys

# layout
from layouts_window import layout_home_window
from layouts_window import layout_image_analysis_window

# button handler
from button_handlers import button_handler_home_window
from button_handlers import button_handler_image_analysis_window


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
        self.connect_handler_to_home_window_button()

    def connect_handler_to_home_window_button(self):
        """
        connect button to handler.
        """
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
        self.connect_handler_to_image_analysis_window_button()
        self.connect_file_model_to_treeview()

        # 格納用変数
        self.data_plot = None

    def connect_handler_to_image_analysis_window_button(self):
        """
        connect button to handler.
        """
        self.pushButton_for_selectDirectory.clicked.connect(lambda: button_handler_image_analysis_window.select_directory(self))
        self.pushButton_for_executeAnalysis.clicked.connect(lambda: button_handler_image_analysis_window.execute(self))
        self.pushButton_for_viewDiagram.clicked.connect(lambda: button_handler_image_analysis_window.view_diagram(self))

    def connect_file_model_to_treeview(self):
        """
        connect file model to treeview.
        """
        # initialize file model
        self.fileModel = QtWidgets.QFileSystemModel()
        self.fileModel.setRootPath(QtCore.QDir.rootPath())

        # QTreeViewにモデルをセット
        self.treeView_for_selectDirectoly.setModel(self.fileModel)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    hw = HomeWindow()
    hw.show()
    sys.exit(app.exec_())
