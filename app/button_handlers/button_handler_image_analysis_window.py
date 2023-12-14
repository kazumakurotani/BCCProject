from PyQt5 import QtWidgets, QtGui, QtCore
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

    # コンボボックスの選択肢を追加
    list_features = [
        "histgrams_features",
        "edge_features",
        "local_features",
        "glcm_features"
    ]
    ImageAnalysisWindow.comboBox_for_selectImage.addItems(file_list)
    ImageAnalysisWindow.comboBox_for_selectFeature.addItems(list_features)

    # 画像情報の取得
    FA = feature_analysis.FeatureAnalysis(list_path_file)

    # 図やグラフ，表のプロットデートを取得
    data_plot = {}

    data_diagrams_histgram_feature = FA.get_diagrams_histgram_features()
    data_plot["diagrams_histgram_feature"] = data_diagrams_histgram_feature

    ImageAnalysisWindow.data_plot = data_plot

    # 解析の終了を通知
    message = "解析終了"
    ImageAnalysisWindow.statusbar.showMessage(message)

def view_diagram(ImageAnalysisWindow):

    data_plot = ImageAnalysisWindow.data_plot

    if data_plot is None:
        # 解析の終了を通知
        message = "解析データが見つかりません．解析を実行してください．"
        ImageAnalysisWindow.statusbar.showMessage(message)

    # 各QGraphicsViewのアドレスを取得
    number_of_views = 10
    list_graphics_view = []
    for i in range(1, number_of_views + 1):
        list_graphics_view.append(getattr(ImageAnalysisWindow, f'graphicsView_tab_{i}'))
    selected_image = ImageAnalysisWindow.comboBox_for_selectImage.currentText()
    selected_feature = ImageAnalysisWindow.comboBox_for_selectFeature.currentText()

    if selected_feature == "histgrams_features":
        selected_data = data_plot["diagrams_histgram_feature"]

        feature_data = selected_data[selected_image]

        for i, color in enumerate(feature_data.keys()):
            data = feature_data[color]
            view_gra = list_graphics_view[i]
            _display_histogram(ImageAnalysisWindow, view_gra, data)

def _display_histogram(ImageAnalysisWindow, graphics_view, cv_img):
    """Display the histogram image in a QGraphicsView, scaled to fit the view."""
    scene = QtWidgets.QGraphicsScene(ImageAnalysisWindow)

    # Convert the OpenCV image to QPixmap
    qimg = QtGui.QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0], QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(qimg)

    # Scale the pixmap to fit the QGraphicsView's size
    scaled_pixmap = pixmap.scaled(graphics_view.width(), graphics_view.height(), QtCore.Qt.KeepAspectRatio)

    # Add the scaled pixmap to the scene and set the scene to the view
    scene.addPixmap(scaled_pixmap)
    graphics_view.setScene(scene)


