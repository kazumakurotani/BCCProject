import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

class ImageDataAugmentor:
    def __init__(self, ):
        """
        ImageDataAugmentorの初期化.
        """
        # 保存先のpath
        self.output_dir_path = 
        self.confirm_dir_path = 

        # Flag

    def create_output_directories(self, labels: list):
        """
        'output'ディレクトリとその下のラベルごとのサブディレクトリを作成する。

        Args:
            labels (list): 作成するラベル名のリスト。
        """
        output_path = os.path.join(self.base_path, 'output')
        self._create_directories(output_path, labels)

    def create_dataset_directories(self, labels: list):
        """
        'dataset/extended'ディレクトリとその下のラベルごとのサブディレクトリを作成する。

        Args:
            labels (list): 作成するラベル名のリスト。
        """
        extended_path = os.path.join(self.base_path, 'dataset', 'extended')
        self._create_directories(extended_path, labels)

    def _create_directories(self, base_path: str, labels: list):
        """
        指定されたベースパスにサブディレクトリを作成する。

        Args:
            base_path (str): サブディレクトリを作成するベースパス。
            labels (list): 作成するラベル名のリスト。
        """
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        for label in labels:
            label_path = os.path.join(base_path, label)
            if not os.path.exists(label_path):
                os.makedirs(label_path)


    def augment_image(self, image):
        # 回転
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # 色調の変更
        hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.6, 1.4)
        color_changed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # ズーム
        zoom_factor = random.uniform(0.5, 1.0)
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        width, height = int(image.shape[1] * zoom_factor), int(image.shape[0] * zoom_factor)
        cropped = color_changed[center_y - height // 2:center_y + height // 2, center_x - width // 2:center_x + width // 2]
        zoomed = cv2.resize(cropped, (image.shape[1], image.shape[0]))

        # シフト
        shift_x, shift_y = random.randint(-20, 20), random.randint(-20, 20)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(zoomed, M, (zoomed.shape[1], zoomed.shape[0]))

        return shifted

    def augment_dataset(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for filename in os.listdir(self.input_dir):
            image_path = os.path.join(self.input_dir, filename)
            image = cv2.imread(image_path)
            
            for i in range(self.augmentation_factor):
                augmented_image = self.augment_image(image)
                cv2.imwrite(os.path.join(self.output_dir, f"aug_{i}_{filename}"), augmented_image)



class DirectoryManager:
    """
    ディレクトリ管理を行うクラス。

    データセットのオリジナルデータと拡張データ用のディレクトリを作成し管理する。

    Attributes:
        base_path (str): ベースとなるパス。
    """

    def __init__(self, base_path: str):
        """
        DirectoryManagerの初期化。

        Args:
            base_path (str): ベースとなるパス。
        """
        self.base_path = base_path

    def create_output_directories(self, labels: list):
        """
        'output'ディレクトリとその下のラベルごとのサブディレクトリを作成する。

        Args:
            labels (list): 作成するラベル名のリスト。
        """
        output_path = os.path.join(self.base_path, 'output')
        self._create_directories(output_path, labels)

    def create_dataset_directories(self, labels: list):
        """
        'dataset/extended'ディレクトリとその下のラベルごとのサブディレクトリを作成する。

        Args:
            labels (list): 作成するラベル名のリスト。
        """
        extended_path = os.path.join(self.base_path, 'dataset', 'extended')
        self._create_directories(extended_path, labels)

    def _create_directories(self, base_path: str, labels: list):
        """
        指定されたベースパスにサブディレクトリを作成する。

        Args:
            base_path (str): サブディレクトリを作成するベースパス。
            labels (list): 作成するラベル名のリスト。
        """
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        for label in labels:
            label_path = os.path.join(base_path, label)
            if not os.path.exists(label_path):
                os.makedirs(label_path)


