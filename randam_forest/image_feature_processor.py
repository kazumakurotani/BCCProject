import cv2
import numpy as np


class FeatureData:
    """
    画像の特徴を機械学習用途で管理するクラス。
    """
    def __init__(self, filepath: str) -> None:
        """
        特徴情報を保持するクラス。

        Args:
            filepath (str): 画像ファイルのパス。
        """
        # 特徴情報
        self.image_preprocessed = None

    def preprocessed(self, ImageData):


    def extract_features(image_data):
        # 引数から画像データを取得
        image = image_data.image_data

        # 画像の前処理：リサイズと平滑化
        resized_image = cv2.resize(image, (300, 300))  # 任意のサイズにリサイズ
        smoothed_image = cv2.GaussianBlur(resized_image, (5, 5), 0)  # ガウシアンブラーを適用

        # 生画像をグレースケールに変換
        gray_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)

        # ヒストグラムを抽出：GRAY，R，G，B，H，S，V
        gray_histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])  # GRAYヒストグラム
        b_histogram = cv2.calcHist([smoothed_image], [0], None, [256], [0, 256])  # Bヒストグラム
        g_histogram = cv2.calcHist([smoothed_image], [1], None, [256], [0, 256])  # Gヒストグラム
        r_histogram = cv2.calcHist([smoothed_image], [2], None, [256], [0, 256])  # Rヒストグラム

        hsv_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2HSV)
        h_histogram = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])  # Hヒストグラム
        s_histogram = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])  # Sヒストグラム
        v_histogram = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])  # Vヒストグラム

        # 各特徴量の情報を記録するためのコメント
        # 0: GRAYヒストグラム, 1: Bヒストグラム, 2: Gヒストグラム, 3: Rヒストグラム
        # 4: Hヒストグラム, 5: Sヒストグラム, 6: Vヒストグラム

        # 特徴行列を作成
        feature_matrix = np.hstack([gray_histogram, b_histogram, g_histogram, r_histogram, h_histogram, s_histogram, v_histogram])

        return feature_matrix
