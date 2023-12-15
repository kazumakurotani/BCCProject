import os
from typing import Dict, List
from tqdm import tqdm

import image_data_prosessor
import image_feature_processor


class RandamForest:

    def __init__(self) -> None:

        # 設定値
        dir_path_dataset = "D:\\github\\BCCProject\\app\\dataset"

        self.dict_image_data, self.list_file_name = self.get_dict_image_data(dir_path_dataset) # ImageDataインスタンスを格納した辞書
        self.dict_features = self.get_dict_features() # ImageFeaturesインスタンスを格納した辞書
        self.feature_matrix = None

    def get_dict_image_data(self, root_dir: str) -> Dict[str, isinstance]:
        """
        データセットのルートディレクトリからすべての画像ファイルのパスを取得し、
        それらのパスに対してImageDataのインスタンスを生成して辞書として返す。

        Args:
            root_dir (str): データセットのルートディレクトリ。

        Returns:
            Dict[str, ImageData]: 各画像ファイルのパスをキーとするImageDataインスタンスの辞書。
        """
        image_paths = self._get_image_file_paths(root_dir)

        # 格納用変数
        dict_image_data = {}
        list_file_name = []

        for path in tqdm(image_paths, desc="Creating ImageData instances"):
            try:
                image_data = image_data_prosessor.ImageData(path)
                image_name = os.path.basename(path)
                list_file_name.append(image_name)
                dict_image_data[image_name] = image_data
            except Exception as e:
                print(f"画像ファイルの処理中にエラーが発生しました: {path}. エラー: {e}")

        return dict_image_data, list_file_name

    def _get_image_file_paths(self, root_dir: str) -> List[str]:
        """
        指定されたルートディレクトリからすべての画像ファイルのパスを取得する。

        Args:
            root_dir (str): 画像ファイルを検索するルートディレクトリ。

        Returns:
            List[str]: すべての画像ファイルのフルパスのリスト。

        Raises:
            ValueError: ルートディレクトリが存在しない、またはディレクトリではない場合に発生。
        """
        if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
            raise ValueError(f"指定されたルートディレクトリは存在しないか、ディレクトリではありません: {root_dir}")

        image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    full_path = os.path.join(subdir, file)
                    image_paths.append(full_path)

        return image_paths

    def get_dict_features(self):

        # 格納用変数
        dict_features = {}

        # 特徴量の取得
        for file_name, image_data in tqdm(self.dict_image_data.items(), desc="Creating ImageData Features"):
            try:
                image = image_data.image_data
                dict_features[file_name] = image_feature_processor.FeatureData(image)
            except Exception as e:
                print(f"画像ファイルの処理中にエラーが発生しました: {file_name}. エラー: {e}")

        return dict_features

    def make_feature_matrix(self, image_data, features):

        # 必要なデータの抽出
        pass

    def main(self):

        # 格納用変数
        feature_matrix = []

        # 必要なデータを呼び出す
        for file_name in tqdm(self.list_file_name, desc="Making Feature Matrix"):
            image_data = self.dict_image_data[file_name]
            features = self.dict_features[file_name]

            feature_matrix = self.make_feature_matrix(feature_matrix, image_data, features)







        # 特徴行列を作成する．
        pass



if __name__ == "__main__":
    rf = RandamForest()
    rf.main()
