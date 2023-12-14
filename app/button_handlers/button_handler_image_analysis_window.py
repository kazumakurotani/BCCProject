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

    # histgramの解析結果の抽出
    data_histgram_featres = ImageData.get_diagrams_histgram_features()


    # 解析の終了を通知
    message = "解析終了"
    ImageAnalysisWindow.statusbar.showMessage(message)

    #     # ヒストグラム画像を取得


    #     # RGBヒストグラムを表示する例
    #     self.labelHistogramRGB.setPixmap(self.convert_cv_qt(histogram_images['RGB']))

    # def get_histogram_images(self):
    #     # ヒストグラム画像を生成または読み込む関数
    #     # ここに、前のステップで作成したヒストグラム画像を生成または読み込むコードを記述
    #     pass

    # def convert_cv_qt(self, cv_img):
    #     """Convert from an opencv image to QPixmap"""
    #     rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    #     h, w, ch = rgb_image.shape
    #     bytes_per_line = ch * w
    #     convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    #     p = convert_to_Qt_format.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
    #     return QtGui.QPixmap.fromImage(p)

