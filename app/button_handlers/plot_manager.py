import matplotlib.pyplot as plt


class PlotManager:
    def __init__(self, style: str):
        # スタイルの適応
        self.style = style
        self._apply_style(self.style)

        # オブジェクトの生成
        self.fig = plt.figure()

    def _apply_style(self, style: str):
        """
        Apply the specified style to the plots based on the current style setting.
        """
        if self.style == 'histgram':
            self._histgram_style()
        elif self.style == 'default':
            self._default_style()
        # 他のスタイルがあればここに追加

    def _histgram_style(self):
        """
        Apply a science paper style to the plots.
        """
        config = {
            'font.family': 'Times New Roman',
            'font.size': 12,

            # axesの設定
            'axes.labelsize': 12,
            'axes.titlesize': 14,

            # 補助目盛の設定
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.direction': 'in',
            'ytick.direction': 'in',

            'savefig.dpi': 300,
            'figure.figsize': [8, 6],

            # 凡例設定
            'legend.fontsize': 12,
            "legend.fancybox": False,
            "legend.framealpha": 1,
            "legend.edgecolor": 'black',
            "legend.markerscale": 5
        }
        plt.rcParams.update(config)

    def _default_style(self):
        """
        Apply a default style to the plots.
        """
        plt.style.use('default')

    def set_title_and_labels(self, title: str, xlabel: str, ylabel: str):
        """
        Set title and labels for the plot.

        Args:
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
