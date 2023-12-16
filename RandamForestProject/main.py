import os
from typing import Dict, List

import image_data_prosessor
import image_feature_processor
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pylab as plt


class RandamForest:

    def __init__(self) -> None:

        # 設定値
        self.dir_path_dataset_desktop = "D:\\github\\BCCProject\\app\\dataset" # 学校のパソコン用のパス
        self.dir_path_dataset_laptop = "C:\\GitHub\\BCCProject\\app\\dataset" # ノートパソコン用のパス

        # 画像データ，特徴
        self.dict_image_data = None # ImageDataインスタンスを格納した辞書
        self.dict_features = None # ImageFeaturesインスタンスを格納した辞書

        self.labels = None # 正解ラベル
        self.feature_matrix = None # 特徴行列
        self.feature_check_dict = {} # 特徴行列の確認

        # フラグ
        self.is_first_call = True
        self.is_laptop_or_desktop = 1 # 1:laptop, 2:desktop

        # 関数の実行
        self.get_dict_image_data()
        self.get_dict_features()

    def get_dict_image_data(self) -> Dict[str, isinstance]:
        """
        データセットのルートディレクトリからすべての画像ファイルのパスを取得し、
        それらのパスに対してImageDataのインスタンスを生成して辞書として返す。

        Args:
            root_dir (str): データセットのルートディレクトリ。

        Returns:
            Dict[str, ImageData]: 各画像ファイルのパスをキーとするImageDataインスタンスの辞書。
        """
        root_dir = self._check_device()

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

        # インスタンスに代入
        self.dict_image_data, self.list_file_name = dict_image_data, list_file_name

    def _check_device(self):
        # 使用デバイスの確認
        if self.is_laptop_or_desktop == 1:
            return self.dir_path_dataset_laptop
        elif self.is_laptop_or_desktop == 2:
            return self.dir_path_dataset_desktop

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
        # パスが有効化の確認
        if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
            raise ValueError(f"指定されたルートディレクトリは存在しないか、ディレクトリではありません: {root_dir}")

        # 画像のファイルパスの取得
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

        self.dict_features = dict_features

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

        # 使用特徴量のリスト作成
        features_dict = {
            "image_gray": features.image_gray,
            "image_saturation": features.image_saturation,
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

        # 目的変数の生成
        labels = []
        for file_name in tqdm(self.list_file_name, desc="Making labels"):
            # 必要なデータを呼び出す
            image_data = self.dict_image_data[file_name]
            label = image_data.label

            labels.append(label)

        # ラベルエンコーダの初期化と変換
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(labels)

        # 特徴行列の生成
        for file_name in tqdm(self.list_file_name, desc="Making Feature Matrix"):
            # 必要なデータを呼び出す
            features = self.dict_features[file_name]

            self.feature_matrix = self.make_feature_matrix(self.feature_matrix, features)

        # 特徴量の情報の表示
        print(sum(self.feature_check_dict.values()))

        # feature_matrix は特徴量行列、labels は正解ラベル
        X_train, X_test, y_train, y_test = train_test_split(self.feature_matrix,
                                                            self.labels,
                                                            test_size=0.2,
                                                            random_state=42)

        rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_classifier.fit(X_train, y_train)

        # テストデータで予測
        y_pred = rf_classifier.predict(X_test)

        # 評価
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # # 特徴量の重要度を取得
        # importances = rf_classifier.feature_importances_

        # # 画像サイズに重要度をリシェイプ
        # importance_map = importances.reshape((128, 128))

        # # ヒートマップの表示
        # plt.imshow(importance_map, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.title("Feature Importances Map")
        # plt.show()


if __name__ == "__main__":
    rf = RandamForest()
    rf.main()
