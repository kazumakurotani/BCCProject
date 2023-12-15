import os
from typing import Dict, List

import image_data_processor
import matplotlib.pylab as plt
import numpy as np
from button_handlers import plot_manager

from io import BytesIO
import cv2

class FeatureAnalysis:
    def __init__(self, list_path_file: List):
        # image dataの取得
        self.dic_instance_image_data = self._get_dic_instance_image_data(list_path_file)

    def _get_dic_instance_image_data(self, list_path_file: List) -> Dict:
        dic_instance_image_data = {}

        for path_file in list_path_file:
            name = os.path.basename(path_file)
            dic_instance_image_data[name] = image_data_processor.ImageData(path_file)

        return dic_instance_image_data

    def get_diagrams_histgram_features(self):
        """
        Plots histograms for each color space.

        Args:
            histograms (Dict[str, np.ndarray]): Dictionary with keys as color spaces
                                                and values as histogram data.
        """
        # ヒストグラムを取得
        dict_histgrams = {}
        for name, instance in self.dic_instance_image_data.items():
            dict_histgrams[name] = instance.histgrams_features

        images_histgram = {}

        for name_file in dict_histgrams.keys():
            dict_images_histgrams = {}
            for color_space, histgrams in dict_histgrams[name_file].items():
                diagram = plot_manager.PlotManager("histgram")
                ax = diagram.fig.add_subplot(111)
                diagram.set_title_and_labels(
                    title = color_space + " " + "histgram",
                    xlabel = "Pixel Intensity",
                    ylabel = "Number of Pixels")

                # グレースケールの場合
                if color_space == 'GRAY':
                    hist = np.array(histgrams.ravel(), np.uint64)
                    ax.plot(hist, color='black')

                elif color_space == 'RGB':
                    colors = {'red': "r", 'green': "g", 'blue':"b"}
                    for i, color in enumerate(colors.keys()):
                        hist = np.array(histgrams[i].ravel(), np.uint64)
                        ax.plot(hist, color=colors[color], label=f'{color}')
                    ax.legend()

                elif color_space == 'HSV':
                    colors = ['hue', 'saturation', 'value']
                    for i, color in enumerate(colors):
                        hist = np.array(histgrams[i].ravel(), np.uint64)
                        ax.plot(hist, label=f'{color}')
                    ax.legend()

                elif color_space == 'YCrCb':
                    colors = ['Y', 'Cr', 'Cb']
                    for i, color in enumerate(colors):
                        hist = np.array(histgrams[i].ravel(), np.uint64)
                        ax.plot(hist, label=f'{color}')
                    ax.legend()

                elif color_space == 'Lab':
                    colors = ['L', 'a', 'b']
                    for i, color in enumerate(colors):
                        hist = np.array(histgrams[i].ravel(), np.uint64)
                        ax.plot(hist, label=f'{color}')
                    ax.legend()

                # Save plot to a BytesIO buffer
                buf = BytesIO()
                plt.savefig(buf, format='png')
                plt.close()

                # Convert buffer to a numpy array
                buf.seek(0)
                img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                buf.close()

                # Convert numpy array to an image
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                dict_images_histgrams[color_space] = img

            images_histgram[name_file] = dict_images_histgrams

        return images_histgram

    def get_diagrams_local_feature(self):
        # 特徴量の取得
        dict_local_feature = {}

        # 特徴量の可視化
        for name, instance in self.dic_instance_image_data.items():
            # 各種データの取得
            image_data = instance.image_preprocessed
            local_feature = instance.local_features
            hog_feature = local_feature["HOG"]
            sift_feature = local_feature["SIFT"]
            lbp_feature = local_feature["HOG"]

            # 特徴量の可視化
            self.visualize_hog(image_data, hog_feature)
            self.visualize_sift(image_data, sift_feature)
            self.visualize_lbp(lbp_feature)

        return None

    def visualize_hog(self, image, hog_features):
        # グレースケールに変換
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # HOGディスクリプタの設定
        cell_size = (8, 8)
        nbins = 9
        n_cells = (gray_image.shape[0] // cell_size[0], gray_image.shape[1] // cell_size[1])

        # HOG特徴量の方向と大きさを表す線分を描画
        hog_image = np.zeros_like(gray_image)
        for i in range(n_cells[0]):
            for j in range(n_cells[1]):
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

    def visualize_sift(self, image, sift_keypoints):
        sift_image = cv2.drawKeypoints(image, sift_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        plt.imshow(cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB))
        plt.title('SIFT features')
        plt.axis('off')
        plt.show()


    def visualize_lbp(self, lbp_image):
        plt.imshow(lbp_image, cmap='gray')
        plt.title('LBP features')
        plt.axis('off')
        plt.show()
