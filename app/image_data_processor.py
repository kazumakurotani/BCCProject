import os
from typing import Dict, Tuple

import cv2
import numpy as np
from skimage import color, feature

import matplotlib.pylab as plt


class ImageData:
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
        self.size = self._get_size(self.image_data)

        # 前処理後の画像
        self.image_preprocessed = self._preprocess_image(self.image_data)
        self.image_converted = self._convert_color_space(self.image_preprocessed)

        # 特徴量
        self.histgrams_features = self._add_histogram_features(self.image_converted)
        self.edge_features = self._add_edge_features(self.image_preprocessed)
        self.local_features = self._add_local_feature(self.image_preprocessed)
        # self.glcm_features = self._add_glcm_features(self.image_preprocessed)

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

    def _get_size(self, image_data: np.ndarray) -> Tuple[int, int]:
        """
        Extracts the size (height and width) of the image data.

        Args:
            image_data (np.ndarray): The image data from which the size is to be extracted.
                                    Expected to be in the format (height, width, channels).

        Returns:
            Tuple[int, int]: A tuple containing the height and width of the image.

        Raises:
            FileNotFoundError: If image_data is not np.ndarray.
            ValueError: If image_data does not have a valid shape for an image.
        """
        if not isinstance(image_data, np.ndarray):
            raise TypeError("image_data must be of type np.ndarray")

        if len(image_data.shape) < 2:
            raise ValueError("image_data does not have a valid shape for an image")

        size = image_data.shape[:2]

        return size

    def _preprocess_image(
            self,
            image_data: np.ndarray,
            resize_dim: tuple = (144, 144),
            kernel_size: tuple = (3,3),
            sigma: int = 2
        ) -> np.ndarray:
        """
        Preprocesses the image by resizing, converting to grayscale, noise reduction, and histogram equalization.

        Args:
            image_path (str): Path to the image file.
            resize_dim (tuple): The dimensions (width, height) to which the image should be resized.

        Returns:
            np.ndarray: The preprocessed image.
        """
        # リサイズ
        image_resized = cv2.resize(image_data, resize_dim)

        # スムージング (e.g., Gaussian Blur)
        image_smoothed = cv2.GaussianBlur(image_resized, kernel_size, sigma)

        return image_smoothed

    def _convert_color_space(self, image_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Converts the given image data into various color spaces and stores them in a dictionary.

        Args:
            image_data (np.ndarray): The original image data in BGR format.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping color space names to their corresponding
                                image data converted into that color space.

        Notes:
            Supported color spaces are GRAY, RGB, HSV, YCrCb, and Lab.
        """
        # 格納用変数
        images_converted = {}

        # 指定色空間
        color_spaces = {
            "GRAY": cv2.COLOR_BGR2GRAY,
            "RGB": cv2.COLOR_BGR2RGB,
            "HSV": cv2.COLOR_BGR2HSV,
            "YCrCb": cv2.COLOR_BGR2YCrCb,
            "Lab": cv2.COLOR_BGR2Lab,
        }

        # 色空間の変換
        for _, color_name in enumerate(color_spaces.keys()):
            color_code = color_spaces[color_name]
            image_converted = cv2.cvtColor(image_data, color_code)

            images_converted[color_name] = image_converted

        print(images_converted.keys())

        return images_converted


    def _add_histogram_features(self, images_converted: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate histograms for each color space provided in images_converted.

        Args:
            images_converted (Dict[str, np.ndarray]): Dictionary with keys as color spaces
                                                    and values as image data in that space.

        Returns:
            Dict[str, np.ndarray]: Dictionary with keys as color spaces and values as histogram data.
        """
        histograms = {}

        # number of bins for histgrams
        bins = 256

        for color_space, image_data in images_converted.items():
            if color_space == 'GRAY':  # gray space have a channel
                hist = cv2.calcHist([image_data], [0], None, [bins], [0, 256])
                histograms[color_space] = hist
            else:
                # if color space is other space, this calculate histgram per channels
                hist_channels = []
                for channel in range(image_data.shape[2]):
                    hist = cv2.calcHist([image_data], [channel], None, [bins], [0, 256])
                    hist_channels.append(hist)
                histograms[color_space] = hist_channels

        return histograms

    def _add_edge_features(self, image_data: np.ndarray, ksize: int = 3) -> Dict[str, np.ndarray]:
        """
        Extracts Sobel edge features from the given image.

        Args:
            image_data (np.ndarray): The original image data in BGR format.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the Sobel edge features.
                                Keys are 'sobel_x' and 'sobel_y' for horizontal and vertical edges, respectively.

        Notes:
            The function applies Sobel edge detection to find horizontal and vertical edges.
            It uses a kernel size of 3 for the Sobel operator.
        """
        # 格納用変数
        edge_features = {}

        # グレイスケールに変換
        image_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        # Sobel Edge Detection for horizontal and vertical edges
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize)

        edge_features["sobel_x"] = sobel_x
        edge_features["sobel_y"] = sobel_y

        return edge_features

    def _add_local_feature(self, image_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts local features (HOG, SIFT, LBP) from the given image using OpenCV and scikit-image.

        Args:
            image_data (np.ndarray): The original image data in BGR format.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the local features.
                                Keys are 'HOG', 'SIFT', and 'LBP' for the respective feature extraction methods.

        Notes:
            - HOG and SIFT features are computed using OpenCV.
            - LBP features are computed using scikit-image.
        """
        local_features = {}

        # グレースケールに変換
        gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        features, hog_img = feature.hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
                                        cells_per_block=(1, 1), visualize=True)
        # matplotlib で表示する。
        plt.figure(figsize=(10, 10))
        plt.imshow(hog_img, cmap='inferno')
        plt.axis('off')
        plt.show()
        plt.savefig('result.png')

        # HOGディスクリプタの設定
        cell_size = (8, 8)
        block_size = (2, 2)
        nbins = 9
        hog = cv2.HOGDescriptor(_winSize=(gray_image.shape[1] // cell_size[1] * cell_size[1],
                                        gray_image.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)

        # HOG特徴量の計算
        hog_features = hog.compute(gray_image)

        # HOG特徴量を画像サイズにリシェイプ
        n_cells = (gray_image.shape[0] // cell_size[0], gray_image.shape[1] // cell_size[1])
        hog_features = hog_features.reshape(n_cells[1] - block_size[1] + 1,
                                            n_cells[0] - block_size[0] + 1,
                                            block_size[0], block_size[1], nbins).transpose((1, 0, 2, 3, 4))

        # HOG特徴量の方向と大きさを表す線分を描画
        hog_image = np.zeros_like(gray_image)
        for i in range(n_cells[0] - block_size[0] + 1):
            for j in range(n_cells[1] - block_size[1] + 1):
                cell_grad = hog_features[i, j, :, :, :]
                cell_grad = cell_grad.transpose((2, 0, 1))
                cell_grad = cell_grad.reshape(-1, nbins)
                max_mag = np.array(cell_grad).max(axis=0)
                ang = np.arange(0, nbins) * (180 / nbins)
                ang_rad = ang * np.pi / 180
                x, y = np.cos(ang_rad) * max_mag, np.sin(ang_rad) * max_mag
                x += j * cell_size[1]
                y += i * cell_size[0]
                for k in range(nbins):
                    pt1 = (int(x[k]), int(y[k]))
                    pt2 = (int(x[k] - np.cos(ang_rad[k]) * max_mag[k]),
                        int(y[k] - np.sin(ang_rad[k]) * max_mag[k]))
                    cv2.line(hog_image, pt1, pt2, int(255 * (k / nbins)))

        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG features')
        plt.axis('off')
        plt.show()


        local_features["HOG"] = hog_features

        # SIFT Feature Extraction
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute(gray_image, None)
        local_features["SIFT"] = descriptors

        # Convert image to grayscale for LBP
        gray_image_lbp = color.rgb2gray(image_data)

        # LBP Feature Extraction
        lbp = feature.local_binary_pattern(gray_image_lbp, P=8, R=1, method="uniform")
        local_features["LBP"] = lbp

        return local_features

    def _add_glcm_features(self, image_data: np.ndarray) -> Dict[str, float]:
        """
        Extracts Gray Level Co-occurrence Matrix (GLCM) features from the given image.

        Args:
            image_data (np.ndarray): The original image data in BGR format.

        Returns:
            Dict[str, float]: A dictionary containing GLCM features such as contrast, correlation,
                            energy, and homogeneity.

        Notes:
            The function converts the image to grayscale and then computes the GLCM.
        """
        # Convert to grayscale
        gray_image = color.rgb2gray(image_data)

        # Normalize and scale the grayscale image
        gray_image = (gray_image * 255).astype("uint8")

        # Compute GLCM
        glcm = feature.greycomatrix(
            gray_image,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True,
        )

        # GLCM properties
        glcm_features = {
            "contrast": feature.greycoprops(glcm, "contrast")[0, 0],
            "correlation": feature.greycoprops(glcm, "correlation")[0, 0],
            "energy": feature.greycoprops(glcm, "energy")[0, 0],
            "homogeneity": feature.greycoprops(glcm, "homogeneity")[0, 0],
        }

        return glcm_features
