from typing import Tuple, Dict, List
import numpy as np

class ImageFeatures:
    """
    Manages the features and information of an image for machine learning purposes.
    """

    def __init__(self, path_file: str) -> None:
        # 画像情報
        self.name = name
        self.path = name
        self.size = size
        self.bgr_data = bgr_data

        # 特徴量
        self.color_features = {}
        self.edge_features = {}
        self.local_features = {}
        self.global_features = {}

    def add_color_feature(self, color_space: str, data: np.ndarray) -> None:
        """
        Adds a color feature in the specified color space to the image.

        Args:
            color_space (str): The color space of the feature (e.g., 'GRAY', 'RGB').
            data (np.ndarray): The color feature data.

        Returns:
            None
        """
        self.color_features[color_space] = data

    def add_edge_feature(self, method: str, data: np.ndarray) -> None:
        """
        Adds an edge feature using the specified method (e.g., 'canny', 'sobel') to the image.

        Args:
            method (str): The method used for edge detection.
            data (np.ndarray): The edge feature data.

        Returns:
            None
        """
        self.edge_features[method] = data

    def add_local_feature(self, descriptor: str, data: np.ndarray) -> None:
        """
        Adds a local feature using the specified descriptor (e.g., 'Hog', 'SIFT') to the image.

        Args:
            descriptor (str): The descriptor used for local feature extraction.
            data (np.ndarray): The local feature data.

        Returns:
            None
        """
        self.local_features[descriptor] = data

    def add_global_feature(self, descriptor: str, data: np.ndarray) -> None:
        """
        Adds a global feature (e.g., 'GIST') to the image.

        Args:
            descriptor (str): The descriptor used for global feature extraction.
            data (np.ndarray): The global feature data.

        Returns:
            None
        """
        self.global_features[descriptor] = data
