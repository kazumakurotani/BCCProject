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

            for color_space, histgrams in dict_histgrams[name_file].items():
                dict_images_histgrams = {}
                diagram = plot_manager.PlotManager("histgram")
                ax = diagram.fig.add_subplot(111)
                diagram.set_title_and_labels(
                    title = color_space + " " + "histgram",
                    xlabel = "Pixel Intensity",
                    ylabel = "Number of Pixels")

                # グレースケールの場合
                if color_space == 'GRAY':
                    ax.hist(histgrams.ravel(), bins=256, color='black', alpha=0.7, density=True)

                elif color_space == 'RGB':
                    colors = ['blue', 'green', 'red']
                    for i, color in enumerate(colors):
                        ax.hist(histgrams[i].ravel(), bins=256, color=color, alpha=0.7, label=f'{color}', density=True)
                    ax.legend()

                elif color_space == 'HSV':
                    colors = ['hue', 'saturation', 'value']
                    for i, color in enumerate(colors):
                        ax.hist(histgrams[i].ravel(), bins=256, alpha=0.7, label=f'{color}', density=True)
                    ax.legend()

                elif color_space == 'YCrCb':
                    colors = ['Y', 'Cr', 'Cb']
                    for i, color in enumerate(colors):
                        ax.hist(histgrams[i].ravel(), bins=256, alpha=0.7, label=f'{color}', density=True)
                    ax.legend()

                elif color_space == 'Lab':
                    colors = ['L', 'a', 'b']
                    for i, color in enumerate(colors):
                        ax.hist(histgrams[i].ravel(), bins=256, alpha=0.7, label=f'{color}', density=True)
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
