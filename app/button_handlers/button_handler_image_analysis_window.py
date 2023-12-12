from PyQt5 import QtWidgets
import os


def select_directory(ImageAnalysisWindow):

    # 参照ディレクトリを取得
    directory = QtWidgets.QFileDialog.getExistingDirectory(ImageAnalysisWindow, "Select Directory")

    # 参照ディレクトリをTree Viewに表示
    if directory:
        # 親ディレクトリのパスを取得
        parent_directory = os.path.dirname(directory)

        # 親ディレクトリをルートとして設定
        ImageAnalysisWindow.fileModel.setRootPath(parent_directory)
        parent_index = ImageAnalysisWindow.fileModel.index(parent_directory)
        ImageAnalysisWindow.treeView_for_selectDirectoly.setRootIndex(parent_index)

        # 選択したディレクトリを展開
        directory_index = ImageAnalysisWindow.fileModel.index(directory)
        ImageAnalysisWindow.treeView_for_selectDirectoly.expand(directory_index)
        ImageAnalysisWindow.treeView_for_selectDirectoly.scrollTo(directory_index)

    else:
        # ディレクトリが正しく選択されていなければその旨を表示
        message = ("ディレクトリが正しく選択されていません．")
        ImageAnalysisWindow.statusbar.showMessage(message)
