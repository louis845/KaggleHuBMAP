import PyQt5
import PyQt5.QtWidgets
import PyQt5.QtCore
import PyQt5.QtGui
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import cv2
import traceback

import model_data_manager

matplotlib.use("qtagg")

image_loader = model_data_manager.get_dataset_dataloader(None)
def construct_wsi(wsi_id):
    wsi_information = model_data_manager.data_information
    wsi_information = wsi_information.loc[wsi_information["source_wsi"] == wsi_id]
    width = wsi_information["i"].max() + 512
    height = wsi_information["j"].max() + 512

    image_np = np.zeros((height, width, 3), dtype=np.uint8)

    for wsi_tile in wsi_information.index:
        try:
            x = wsi_information.loc[wsi_tile, "i"]
            y = wsi_information.loc[wsi_tile, "j"]
            image_np[y:y+512, x:x+512, :] = image_loader.get_image_data(wsi_tile)
        except Exception as e:
            traceback.print_exc()

    print("WSI:", wsi_id)
    print(image_np.dtype)
    print(image_np.shape)
    print(image_loader.get_image_data(wsi_information.index[0]).dtype)
    print(image_loader.get_image_data(wsi_information.index[0]).shape)
    print("---------------------")

    return image_np


class MainWindow(PyQt5.QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Set size to 1920x1080
        self.resize(1920, 1080)

        # Create the tab widget and add the tabs
        self.tab_widget = PyQt5.QtWidgets.QTabWidget()
        self.tab_widget.addTab(self.create_tab(1), "WSI_1")
        self.tab_widget.addTab(self.create_tab(2), "WSI_2")
        """self.tab_widget.addTab(self.create_tab(3), "WSI_3")
        self.tab_widget.addTab(self.create_tab(4), "WSI_4")
        self.tab_widget.addTab(self.create_tab(6), "WSI_6")
        self.tab_widget.addTab(self.create_tab(7), "WSI_7")
        self.tab_widget.addTab(self.create_tab(8), "WSI_8")
        self.tab_widget.addTab(self.create_tab(9), "WSI_9")
        self.tab_widget.addTab(self.create_tab(10), "WSI_10")
        self.tab_widget.addTab(self.create_tab(11), "WSI_11")
        self.tab_widget.addTab(self.create_tab(12), "WSI_12")
        self.tab_widget.addTab(self.create_tab(13), "WSI_13")
        self.tab_widget.addTab(self.create_tab(14), "WSI_14")"""

        # Create the layout and add the tab widget and the button
        layout = PyQt5.QtWidgets.QVBoxLayout()
        layout.addWidget(self.tab_widget)

        # Create the widget and set the layout
        widget = PyQt5.QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)
        self.setWindowTitle("Tabbed Plotting")

    def create_tab(self, wsi_id):
        try:
            wsi_np_img = construct_wsi(wsi_id)
            # Create a QWidget that contains a matplotlib navigation toolbar and canvas, which plots the wsi_np_img
            tab = PyQt5.QtWidgets.QWidget()
            fig = plt.figure()

            plt.imshow(wsi_np_img)

            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(PyQt5.QtWidgets.QSizePolicy.Expanding, PyQt5.QtWidgets.QSizePolicy.Expanding)

            toolbar = NavigationToolbar(canvas, tab)

            layout = PyQt5.QtWidgets.QVBoxLayout()
            layout.addWidget(canvas)
            layout.addWidget(toolbar)

            tab.setLayout(layout)
        except Exception as e:
            traceback.print_exc()

        return tab


if __name__ == "__main__":
    app = PyQt5.QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()