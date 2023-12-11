from typing import Tuple, Dict, List
import numpy as np
import os
import cv2


class ImageData():
    """
    Manages the features and information of an image for machine learning purposes.
    """

    def __init__(self, filepath: str) -> None:
        """
        Class that holds image information

        Args:
            path_file (str): _description_
        """
        # 画像情報
        self.filepath = filepath
        self.name = self._get_filename(filepath)
        self.image_data = self._get_image_data(filepath)
        self.size = self.data_file.shape[:2]

        # 特徴量
        self.color_features = {}
        self.edge_features = {}
        self.local_features = {}
        self.global_features = {}


    def _get_filename(self, filepath: str) -> str:
        """
        get filename

        Args:
            filepath (str): fulll path or image

        Returns:
            str: The filename extracted from the given path.

        Raises:
            ValueError: If the filepath is empty or None.
        """
        if not filepath:
            raise ValueError("Filepath cannot be empty")

        filename = os.path.splitext(os.path.basename(filepath))[0]

        return filename


    def _get_image_data(self, filepath: str) -> np.ndarray:
        """
        Reads an image from a given file path and returns its data in BGR format.

        Args:
            filepath (str): The file path of the image to be read.

        Returns:
            np.ndarray: The image data in BGR format.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid image or cannot be read as an image.
        """
        # Check if the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read the image
        image_data = cv2.imread(filepath, cv2.IMREAD_COLOR)

        # Check if the image has been correctly read
        if image_data is None:
            raise ValueError(f"Failed to read the file as an image: {filepath}")

        return image_data


    def add_color_feature(self, color_space: str, data: np.ndarray) -> None:

        self.color_features[color_space] = data


    def add_edge_feature(self, method: str, data: np.ndarray) -> None:

        self.edge_features[method] = data


    def add_local_feature(self, descriptor: str, data: np.ndarray) -> None:

        self.local_features[descriptor] = data


    def add_global_feature(self, descriptor: str, data: np.ndarray) -> None:

        self.global_features[descriptor] = data
