import os
from typing import Dict, List
from tqdm import tqdm

import image_data_prosessor
import image_feature_processor

import numpy as np


class RandamForest:

    def __init__(self) -> None:

        # 設定値
        # dir_path_dataset_desktop = "D:\\github\\BCCProject\\app\\dataset"
        dir_path_dataset_laptop = "C:\\GitHub\\BCCProject\\app\\dataset"

        self.dict_image_data, self.list_file_name = self.get_dict_image_data(dir_path_dataset_laptop) # ImageDataインスタンスを格納した辞書
        self.dict_features = self.get_dict_features() # ImageFeaturesインスタンスを格納した辞書
        self.feature_matrix = None
        self.feature_check_dict = {}

        # フラグ
        self.is_first_call = True

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

    def make_feature_matrix(self, feature_matrix, features):
        """
        与えられた特徴量から特徴量行列を作成する。

        Args:
            feature_matrix: 既存の特徴量行列（存在する場合）。
            features: 抽出した特徴量を含むオブジェクト。

        Returns:
            np.ndarray: 特徴量行列。
        """
        # 格納用変数
        combined_features = None
        feature_check_dict = {}

        # 使用特徴量のリスト作成
        features_dict = {
            "image_gray": features.image_gray,
            "gray_histogram": features.gray_histogram,
            "blue_histogram": features.blue_histogram,
            "green_histogram": features.green_histogram,
            "red_histogram": features.red_histogram,
            "hue_histogram": features.hue_histogram,
            "saturation_histogram": features.saturation_histogram,
            "value_histogram": features.value_histogram
        }

        # 特徴量の結合
        for feature_name, feature_data in features_dict.items():
            data = feature_data.flatten()
            if combined_features is not None:
                combined_features = np.concatenate([combined_features, data])
            else:
                combined_features = data

            if self.is_first_call is True:
                self.feature_check_dict[feature_name] = len(data)

        # 特徴量行列の作成
        if feature_matrix is not None:
            feature_matrix = np.vstack([feature_matrix, combined_features])
        else:
            feature_matrix = combined_features
            self.is_first_call = False

        return feature_matrix

    def main(self):
        for file_name in tqdm(self.list_file_name, desc="Making Feature Matrix"):
            # 必要なデータを呼び出す
            image_data = self.dict_image_data[file_name]
            features = self.dict_features[file_name]

            # 特徴行列の作成
            self.feature_matrix = self.make_feature_matrix(self.feature_matrix, features)

        # 特徴量の情報の表示
        print(sum(self.feature_check_dict.values()))

        pass

if __name__ == "__main__":
    rf = RandamForest()
    rf.main()
