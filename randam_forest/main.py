import os
from typing import Dict, List
from tqdm import tqdm

import Image_data_prosessor


class RandamForest:

    def __init__(self) -> None:

        # 設定値
        dir_path_dataset = "D:\\github\\BCCProject\\app\\dataset"

        self.dict_image_data = self.get_dict_image_data(dir_path_dataset) # ImageDataインスタンスを格納した辞書
        self.dict_features = {} # ImageFeaturesインスタンスを格納した辞書

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
        dict_image_data = {}
        for path in tqdm(image_paths, desc="Creating ImageData instances"):
            try:
                image_data = Image_data_prosessor.ImageData(path)
                image_name = os.path.basename(path)
                dict_image_data[image_name] = image_data
            except Exception as e:
                print(f"画像ファイルの処理中にエラーが発生しました: {path}. エラー: {e}")

        return dict_image_data

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


    def main(self):
        pass


if __name__ == "__main__":
    rf = RandamForest()
    rf.main()
