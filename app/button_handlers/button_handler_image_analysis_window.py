from PyQt5 import QtWidgets
import os
from button_handlers import feature_analysis


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

def execute(ImageAnalysisWindow):

    # 解析の開始を通知
    message = "解析中..."
    ImageAnalysisWindow.statusbar.showMessage(message)

    # Tree Viewの参照先のモデルを作成
    viewModel = ImageAnalysisWindow.treeView_for_selectDirectoly.model()

    # 参照ディレクトリのパスを取得
    selected_indexes = ImageAnalysisWindow.treeView_for_selectDirectoly.selectedIndexes()

    if not selected_indexes:
        message = "ディレクトリが正しく選択されていません."
        ImageAnalysisWindow.statusbar.showMessage(message)
        return None

    # 解析ディレクトリのパス取得
    path_dir = viewModel.filePath(selected_indexes[0])
    file_list = os.listdir(path_dir)

    # 解析ファイルのパス取得
    list_path_file = []
    for f in file_list:
        path_file = os.path.join(path_dir, f)
        list_path_file.append(path_file)

    # 画像情報の取得
    ImageData = feature_analysis.FeatureAnalysis(list_path_file)
    ImageData.get_histograms_histgram_features()

    # 解析の終了を通知
    message = "解析終了"
    ImageAnalysisWindow.statusbar.showMessage(message)


