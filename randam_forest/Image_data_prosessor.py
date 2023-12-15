import os
from typing import Tuple

import cv2
import numpy as np


class ImageData:
    """
    画像の特徴と情報を機械学習用途で管理するクラス。
    """

    def __init__(self, filepath: str) -> None:
        """
        画像情報を保持するクラス。

        Args:
            filepath (str): 画像ファイルのパス。
        """
        # 画像情報
        self.filepath = filepath
        self.name = self._get_filename(filepath)
        self.label = self._get_label(filepath)
        self.image_data = self._get_image_data(filepath)
        self.size = self._get_size(self.image_data)

    def _get_filename(self, filepath: str) -> str:
        """
        与えられたファイルパスからファイル名を取得する。

        Args:
            filepath (str): 画像ファイルの完全なパス。

        Returns:
            str: ファイルパスから抽出したファイル名。

        Raises:
            ValueError: ファイルパスが空またはNoneの場合に発生。
        """
        if not filepath:
            raise ValueError("ファイルパスは空にできません")

        filename = os.path.splitext(os.path.basename(filepath))[0]

        return filename

    def _get_label(self, file_path: str) -> str:
        """
        与えられたファイルパスからディレクトリ名を取得する。

        Args:
            file_path (str): ファイルのパス。

        Returns:
            str: ファイルが含まれているディレクトリの名前。

        Raises:
            ValueError: ファイルパスが空またはNoneの場合に発生。
        """
        if not file_path:
            raise ValueError("ファイルパスは空にできません")

        directory_path = os.path.dirname(file_path)
        directory_name = os.path.basename(directory_path)

        return directory_name

    def _get_image_data(self, filepath: str) -> np.ndarray:
        """
        与えられたファイルパスから画像を読み込み、BGR形式でそのデータを返す。

        Args:
            filepath (str): 読み込まれる画像のファイルパス。

        Returns:
            np.ndarray: BGR形式の画像データ。

        Raises:
            FileNotFoundError: ファイルが存在しない場合に発生。
            ValueError: ファイルが有効な画像として読み込めない場合に発生。
        """
        # ファイルの存在を確認
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")

        # 画像を読み込む
        image_data = cv2.imread(filepath, cv2.IMREAD_COLOR)

        # 画像が正しく読み込まれたかを確認
        if image_data is None:
            raise ValueError(f"画像ファイルとして読み込めませんでした: {filepath}")

        return image_data

    def _get_size(self, image_data: np.ndarray) -> Tuple[int, int]:
        """
        画像データからサイズ（高さと幅）を抽出する。

        Args:
            image_data (np.ndarray): サイズを抽出する画像データ。
                                     形式は(height, width, channels)とする。

        Returns:
            Tuple[int, int]: 画像の高さと幅を含むタプル。

        Raises:
            TypeError: image_dataがnp.ndarray型でない場合に発生。
            ValueError: image_dataが有効な画像の形状を持たない場合に発生。
        """
        if not isinstance(image_data, np.ndarray):
            raise TypeError("image_dataはnp.ndarray型である必要があります")

        if len(image_data.shape) < 2:
            raise ValueError("image_dataは有効な画像の形状を持っていません")

        size = image_data.shape[:2]

        return size
