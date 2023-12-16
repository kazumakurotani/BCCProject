import cv2
import numpy as np
from typing import Dict


class FeatureData:
    """
    画像の特徴を機械学習用途で管理するクラス。
    """
    def __init__(self, image: np.ndarray) -> None:
        """
        特徴情報を保持するクラス。

        Args:
            filepath (str): 画像ファイルのパス。
        """
        # 格納用変数
        self.image_preprocessed = None

        self.image_gray = None

        self.image_hue = None
        self.image_saturation = None
        self.image_value = None

        self.gray_histogram = None
        self.blue_histogram = None
        self.green_histogram = None
        self.red_histogram = None
        self.hue_histogram = None
        self.saturation_histogram = None
        self.value_histogram = None

        # 特徴抽出
        self.preprocessed(image)
        self.get_image_gray(self.image_preprocessed)
        self.get_image_saturation(self.image_preprocessed)
        self.get_color_histgrams(self.image_preprocessed)

    def preprocessed(self, image: np.ndarray) -> np.ndarray:
        """
        画像に対してガウシアンブラーによる平滑化とリサイズを行う。

        ガウシアンフィルタを適用して画像を平滑化し、指定されたサイズにリサイズする。

        Args:
            image (np.ndarray): 処理する画像。BGRカラースペースであることが想定される。

        Returns:
            np.ndarray: 平滑化およびリサイズされた画像。
        """
        # パラメータの設定
        size_resize = (128, 128)
        kernel_size_smooth = (3, 3)
        sigma_smooth = 1

        # 画像の前処理: ガウシアンブラーによる平滑化、リサイズ
        smoothed_image = cv2.GaussianBlur(image, kernel_size_smooth, sigma_smooth)
        resized_image = cv2.resize(smoothed_image, size_resize)

        # 変数に代入
        self.image_preprocessed = resized_image


    def get_image_gray(self, image: np.ndarray) -> np.ndarray:
        """
        BGRカラースペースの画像をグレースケールに変換する。

        Args:
            image (np.ndarray): グレースケールに変換する画像。BGRカラースペースであることが想定される。

        Returns:
            np.ndarray: グレースケールに変換された画像。
        """
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image_gray = image_gray / 255

    def get_image_saturation(self, image: np.ndarray) -> np.ndarray:
        """
        画像を彩度のみを抽出する変換する。

        BGRカラースペースの画像をHSVカラーベースに変換し,彩度のみを抽出する.

        Args:
            image (np.ndarray): グレースケールに変換する画像。BGRカラースペースであることが想定される。

        Returns:
            np.ndarray: グレースケールに変換された画像。
        """
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        self.image_hue = image_hsv[:,:,0] / 180
        self.image_saturation = image_hsv[:,:,1] / 255
        self.image_value = image_hsv[:,:,2] / 255

    def get_color_histgrams(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        画像からGRAY, RGB, HSVカラースペースのヒストグラムを抽出し、
        それらを辞書に格納して返す。

        Args:
            image (np.ndarray): cv2.imreadで読み込まれた画像データ。

        Returns:
            Dict[str, np.ndarray]: 各色空間のヒストグラムを含む辞書。
        """
        # グレースケールに変換
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 画素数の算出
        height, width = gray_image.shape[:2]
        number_of_pixels = height * width

        # パラメータ
        bins = 64

        # ヒストグラムの抽出
        self.gray_histogram = cv2.calcHist([gray_image], [0], None, [bins], [0, 256]) / number_of_pixels
        self.blue_histogram = cv2.calcHist([image], [0], None, [bins], [0, 256]) / number_of_pixels
        self.green_histogram = cv2.calcHist([image], [1], None, [bins], [0, 256]) / number_of_pixels
        self.red_histogram = cv2.calcHist([image], [2], None, [bins], [0, 256]) / number_of_pixels

        # HSVに変換してヒストグラムの抽出
        self.hue_histogram = cv2.calcHist([hsv_image], [0], None, [bins], [0, 256]) / number_of_pixels
        self.saturation_histogram = cv2.calcHist([hsv_image], [1], None, [bins], [0, 256]) / number_of_pixels
        self.value_histogram = cv2.calcHist([hsv_image], [2], None, [bins], [0, 256]) / number_of_pixels
