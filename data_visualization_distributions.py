import os.path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QVBoxLayout, QWidget, QTabWidget, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

"""Create a QT GUI, containing 14 tabs. In each tab, load "hsv_distributions/wsi_<>.npz" and plot the distributions with matplotlib.
The plot should contain 3 bar graphs separately, each of which represents the distribution of H, S, V channel of the image."""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set size to 1920x1080
        self.resize(1920, 1080)

        # Create the tab widget and add the tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.create_tab(1), "WSI_1")
        self.tab_widget.addTab(self.create_tab(2), "WSI_2")
        self.tab_widget.addTab(self.create_tab(3), "WSI_3")
        self.tab_widget.addTab(self.create_tab(4), "WSI_4")
        self.tab_widget.addTab(self.create_tab(6), "WSI_6")
        self.tab_widget.addTab(self.create_tab(7), "WSI_7")
        self.tab_widget.addTab(self.create_tab(8), "WSI_8")
        self.tab_widget.addTab(self.create_tab(9), "WSI_9")
        self.tab_widget.addTab(self.create_tab(10), "WSI_10")
        self.tab_widget.addTab(self.create_tab(11), "WSI_11")
        self.tab_widget.addTab(self.create_tab(12), "WSI_12")
        self.tab_widget.addTab(self.create_tab(13), "WSI_13")
        self.tab_widget.addTab(self.create_tab(14), "WSI_14")
        self.tab_widget.addTab(self.create_segmentation_tab(), "Segmentation")

        # Create the layout and add the tab widget and the button
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)

        # Create the widget and set the layout
        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)
        self.setWindowTitle("Tabbed Plotting")

    def create_tab(self, wsi_id):
        # Create a figure with 3 vertical subplots
        figure, axes = plt.subplots(3, 1, figsize=(5, 4))

        # Generate the data
        data = np.load("hsv_distributions/wsi_{}.npz".format(wsi_id))
        h = data["hue"]
        s = data["saturation"]
        v = data["value"]
        x = np.arange(0, 256, 1)

        # Plot the data
        axes[0].bar(x, h, color="red")
        axes[1].bar(x, s, color="green")
        axes[2].bar(x, v, color="blue")

        # Set the title
        axes[0].set_title("WSI_{} Hue".format(wsi_id))
        axes[1].set_title("WSI_{} Saturation".format(wsi_id))
        axes[2].set_title("WSI_{} Value".format(wsi_id))

        # Create the canvas and add it to the layout
        canvas = FigureCanvas(figure)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create the toolbar and add it to the layout
        toolbar = NavigationToolbar(canvas, self)

        # Create a QWidget and layout
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        layout.addWidget(toolbar)
        widget = QWidget()
        widget.setLayout(layout)

        return widget

    def create_segmentation_tab(self):
        if os.path.isdir("segmentation_data"):
            # Load the data from segmentation_data/statistics.npz. There are three arrays in the file: "glomerulus_size", "blood_vessel_size", "unknown_size".
            seg_statistics = np.load("segmentation_data/statistics.npz")

            # Create a figure with 3 vertical subplots
            figure, axes = plt.subplots(3, 1, figsize=(5, 4))

            # Plot the data with seaborn kdeplot
            sns.kdeplot(seg_statistics["glomerulus_size"], ax=axes[0], color="red")
            sns.kdeplot(seg_statistics["blood_vessel_size"], ax=axes[1], color="green")
            sns.kdeplot(seg_statistics["unknown_size"], ax=axes[2], color="blue")

            # Set the title
            axes[0].set_title("Glomerulus Size")
            axes[1].set_title("Blood Vessel Size")
            axes[2].set_title("Unknown Size")

            # Plot the median, max, min, 5 and 95 percentile values as vertical lines
            axes[0].axvline(np.median(seg_statistics["glomerulus_size"]), color="red", linestyle="--")
            axes[0].axvline(seg_statistics["glomerulus_size"].max(), color="red", linestyle=":")
            axes[0].axvline(seg_statistics["glomerulus_size"].min(), color="red", linestyle=":")
            axes[0].axvline(np.percentile(seg_statistics["glomerulus_size"], 5), color="red", linestyle="-")
            axes[0].axvline(np.percentile(seg_statistics["glomerulus_size"], 95), color="red", linestyle="-")

            axes[1].axvline(np.median(seg_statistics["blood_vessel_size"]), color="green", linestyle="--")
            axes[1].axvline(seg_statistics["blood_vessel_size"].max(), color="green", linestyle=":")
            axes[1].axvline(seg_statistics["blood_vessel_size"].min(), color="green", linestyle=":")
            axes[1].axvline(np.percentile(seg_statistics["blood_vessel_size"], 5), color="green", linestyle="-")
            axes[1].axvline(np.percentile(seg_statistics["blood_vessel_size"], 95), color="green", linestyle="-")

            axes[2].axvline(np.median(seg_statistics["unknown_size"]), color="blue", linestyle="--")
            axes[2].axvline(seg_statistics["unknown_size"].max(), color="blue", linestyle=":")
            axes[2].axvline(seg_statistics["unknown_size"].min(), color="blue", linestyle=":")
            axes[2].axvline(np.percentile(seg_statistics["unknown_size"], 5), color="blue", linestyle="-")
            axes[2].axvline(np.percentile(seg_statistics["unknown_size"], 95), color="blue", linestyle="-")

            # Create the canvas and add it to the layout
            canvas = FigureCanvas(figure)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Create the toolbar and add it to the layout
            toolbar = NavigationToolbar(canvas, self)

            # Create a QWidget and layout
            layout = QVBoxLayout()
            layout.addWidget(canvas)
            layout.addWidget(toolbar)

            widget = QWidget()
            widget.setLayout(layout)

            return widget





if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()