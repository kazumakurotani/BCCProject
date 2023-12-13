import matplotlib.pyplot as plt

class PlotStyle:
    def __init__(self):
        self.styles = {
            'science': self._science_style,
            'default': self._default_style,
            # 他のスタイルを追加可能
        }

    def apply_style(self, style_name: str):
        """
        Apply the specified style to the plots.

        Args:
            style_name (str): The name of the style to apply.
        """
        style_func = self.styles.get(style_name, self._default_style)
        style_func()

    def _science_style(self):
        """
        Apply a science paper style to the plots.
        """
        plt.style.use('seaborn-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 12,
            'figure.titlesize': 16,
            'savefig.dpi': 300,
            'figure.figsize': [8, 6]
        })

    def _default_style(self):
        """
        Apply a default style to the plots.
        """
        plt.style.use('default')

