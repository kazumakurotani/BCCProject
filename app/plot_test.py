from button_handlers import plot_diagram_for_feature_analysis
import numpy as np
import matplotlib.pylab as plt


x = np.linspace(0, 1, 100)
y = x ** 2
test = plot_diagram_for_feature_analysis.PlotDiagramManager("histgram")

ax = test.fig.add_subplot(111)
ax.plot(x, y)
test.set_title_and_labels(title = "TEST",  xlabel = "x", ylabel = "y")

plt.show()


