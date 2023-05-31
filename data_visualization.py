"""Python file to visualize the data. This program contains a main GUI window, with Qt PyQt5."""


import os
import json
import PyQt5
import PyQt5.QtWidgets
import PyQt5.QtCore

import pandas as pd
import numpy as np

import cv2
import sklearn.cluster

import config

import model_data_manager

with open(os.path.join(config.input_data_path, "polygons.jsonl")) as json_file:
    json_list = list(json_file)

all_polygon_masks = {}
for json_str in json_list:
    polygon_masks = json.loads(json_str)
    all_polygon_masks[polygon_masks["id"]] = polygon_masks["annotations"]


class MainWindow(PyQt5.QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Visualization")
        self.resize(1920, 1080)
        self.unique_dataset_values = model_data_manager.data_information["dataset"].value_counts().sort_index()
        self.unique_wsi_values = model_data_manager.data_information["source_wsi"].value_counts().sort_index()

        # Create the left selection area on the left, and the tabbed interface on the right. The left selection area should have 200 width, and the tabbed interface should have 1720 width.
        # The left selection area is a QWidget, and the tabbed interface is a QTabWidget.
        self.left_selection_area = PyQt5.QtWidgets.QWidget(self)
        self.left_selection_area.setGeometry(0, 0, 300, 1080)
        self.right_area = PyQt5.QtWidgets.QWidget(self)
        self.right_area.setGeometry(300, 0, 1620, 1080)
        self.right_area_layout = PyQt5.QtWidgets.QVBoxLayout(self.right_area)

        self.comparison_dropdown = PyQt5.QtWidgets.QComboBox(self.right_area)
        self.comparison_segmentation_dropdown = PyQt5.QtWidgets.QComboBox(self.right_area)
        self.tabbed_interface = PyQt5.QtWidgets.QTabWidget(self.right_area)
        self.custom_algorithm_button = PyQt5.QtWidgets.QPushButton(self.right_area,
                                                                   text="Run Custom Image Algorithm (written in Python)")

        self.right_area_layout.addWidget(self.comparison_dropdown)
        self.right_area_layout.addWidget(self.comparison_segmentation_dropdown)
        self.right_area_layout.addWidget(self.tabbed_interface)
        self.right_area_layout.addWidget(self.custom_algorithm_button)


        self.total_tabs = []

        # Create the list of checkboxes indicating which datasets to be shown, and the list of buttons to show the data visualizations. First split the left selection area into two vertical halves, one for the list of checkboxes and one for the list of buttons.
        # The list of checkboxes is a QWidget, and the list of buttons is a QScrollArea.
        self.checkboxes_widget = PyQt5.QtWidgets.QWidget(self.left_selection_area)
        self.checkboxes_widget.setGeometry(0, 0, 150, 1080)
        self.buttons_scroll_area = PyQt5.QtWidgets.QScrollArea(self.left_selection_area)
        self.buttons_scroll_area.setGeometry(150, 0, 150, 1080)
        self.buttons_scroll_area.setVerticalScrollBarPolicy(PyQt5.QtCore.Qt.ScrollBarAlwaysOn)
        self.buttons_scroll_area.setHorizontalScrollBarPolicy(PyQt5.QtCore.Qt.ScrollBarAlwaysOff)
        self.buttons_scroll_area.setWidgetResizable(True)
        self.buttons_widget = PyQt5.QtWidgets.QWidget(self.buttons_scroll_area)
        self.buttons_scroll_area.setWidget(self.buttons_widget)

        # In the checkboxes_widget, there should be two subwidgets named datasets_widget and wsi_widget, stacked vertically.
        # In the datasets_widget, there should be a list of checkboxes, one for each dataset. The checkboxes should be checked by default.
        # In the wsi_widget, there should be a list of checkboxes, one for each WSI. The checkboxes should be checked by default.
        self.checkboxes_layout = PyQt5.QtWidgets.QVBoxLayout(self.checkboxes_widget)
        self.datasets_widget = PyQt5.QtWidgets.QWidget(self.checkboxes_widget)
        self.wsi_widget = PyQt5.QtWidgets.QWidget(self.checkboxes_widget)
        self.checkboxes_layout.addWidget(self.datasets_widget)
        self.checkboxes_layout.addWidget(self.wsi_widget)

        # Add the dataset checkboxes into the list of checkboxes. The checkboxes are contained inside a QVBoxLayout.
        self.datasets_layout = PyQt5.QtWidgets.QVBoxLayout(self.datasets_widget)
        self.dataset_checkboxes = {}
        for dataset_group in self.unique_dataset_values.index:
            dataset_group_str = str(dataset_group)
            checkbox = PyQt5.QtWidgets.QCheckBox(dataset_group_str + " (" + str(self.unique_dataset_values[dataset_group]) + ")", parent=self.datasets_widget)
            checkbox.setChecked(True)
            self.dataset_checkboxes[dataset_group] = checkbox
            self.datasets_layout.addWidget(checkbox)

        # Add the WSI checkboxes into the list of checkboxes. The checkboxes are contained inside a QVBoxLayout.
        self.wsi_layout = PyQt5.QtWidgets.QVBoxLayout(self.wsi_widget)
        self.wsi_checkboxes = {}
        for wsi in self.unique_wsi_values.index:
            wsi_str = str(wsi)
            checkbox = PyQt5.QtWidgets.QCheckBox(wsi_str + " (" + str(self.unique_wsi_values[wsi]) + ")", parent=self.wsi_widget)
            checkbox.setChecked(True)
            self.wsi_checkboxes[wsi] = checkbox
            self.wsi_layout.addWidget(checkbox)


        # Add the buttons into the list of buttons. The buttons are contained inside a QVBoxLayout.
        self.buttons_layout = PyQt5.QtWidgets.QVBoxLayout(self.buttons_widget)
        self.buttons = []

        for entry in model_data_manager.data_information.index:
            button = PyQt5.QtWidgets.QPushButton(text=str(entry), parent=self.buttons_widget)
            self.buttons.append(button)
            self.buttons_layout.addWidget(button)

        # Connect the checkbox signals to the checkbox_clicked function.
        for checkbox in self.dataset_checkboxes.values():
            checkbox.clicked.connect(self.checkbox_clicked)
        for checkbox in self.wsi_checkboxes.values():
            checkbox.clicked.connect(self.checkbox_clicked)

        # Connect the button signals to the selection_button_clicked function.
        for button in self.buttons:
            button.clicked.connect(lambda checked, clicked_data_entry=button.text(): self.selection_button_clicked(clicked_data_entry))

        self.custom_algorithm_button.clicked.connect(self.custom_algorithm_button_clicked)

        # Loop through the existing datasets, set the options for the comparison dropdown.
        for dataset in model_data_manager.list_datasets():
            self.comparison_dropdown.addItem(dataset)

        # Loop through the existing datasets, set the options for the comparison segmentation dropdown.
        self.comparison_segmentation_dropdown.addItem("None")
        for dataset in model_data_manager.list_datasets():
            self.comparison_segmentation_dropdown.addItem(dataset)




    def checkbox_clicked(self):
        # Loop through all the buttons, and hide/show the buttons depending on which checkboxes are checked.
        for button in self.buttons:
            button_dataset = model_data_manager.data_information.loc[button.text(), "dataset"]
            visible = self.dataset_checkboxes[button_dataset].isChecked() and self.wsi_checkboxes[model_data_manager.data_information.loc[button.text(), "source_wsi"]].isChecked()
            button.setVisible(visible)

    def selection_button_clicked(self, clicked_data_entry):
        # Create a new tab in the tabbed interface, and add the data visualization for the selected dataset into the new tab.
        # Load an image from os.path.join(config.input_data_path, "train", clicked_data_entry + ".tif") using cv2, and save it as a numpy array.
        # The image is to be displayed in a Qt widget, so it needs to be converted into a QImage first.

        # Load the image here
        image = cv2.imread(os.path.join(config.input_data_path, "train", clicked_data_entry + ".tif"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Add the polygon annotations if it exists
        if clicked_data_entry in all_polygon_masks:
            for polygon_mask in all_polygon_masks[clicked_data_entry]:
                # The color depends on the type, default unknown color = blue
                color = (0, 0, 255)
                if polygon_mask["type"] == "glomerulus":
                    color = (0, 255, 0) # green
                elif polygon_mask["type"] == "blood_vessel":
                    color = (255, 0, 0) # red

                # Draw the polygon
                polygon_coordinate_list = polygon_mask["coordinates"][0] # This is a list of integer 2-tuples, representing the coordinates.
                for i in range(len(polygon_coordinate_list)):
                    cv2.line(image, polygon_coordinate_list[i], polygon_coordinate_list[(i + 1) % len(polygon_coordinate_list)], color, 3)

                # Fill the polygon with the color, with 35% opacity
                overlay = image.copy()
                cv2.fillPoly(overlay, [np.array(polygon_coordinate_list)], color)
                image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

        widget = PyQt5.QtWidgets.QWidget()
        layout = PyQt5.QtWidgets.QVBoxLayout()
        widget.setLayout(layout)

        label = self.create_label_from_image(image, widget)
        alt_dataset = str(self.comparison_dropdown.currentText())
        segmentation_dataset = str(self.comparison_segmentation_dropdown.currentText())

        if model_data_manager.dataset_exists(alt_dataset):
            data_loader = model_data_manager.get_dataset_dataloader(alt_dataset)
            image_transformed = np.array(data_loader.get_image_data(clicked_data_entry))
            data_loader.close()
            del data_loader

            # Add the segmentation mask as a white overlay if it exists
            if segmentation_dataset != "None" and model_data_manager.dataset_exists(segmentation_dataset):
                data_loader = model_data_manager.get_dataset_dataloader(segmentation_dataset)
                segmentation_mask = np.array(data_loader.get_image_data(clicked_data_entry))
                data_loader.close()
                del data_loader

                if len(segmentation_mask.shape) == 2:
                    segmentation_mask = np.repeat(np.expand_dims(segmentation_mask, axis=2), axis=2, repeats=3)

                image_transformed = cv2.addWeighted(image_transformed, 0.5, segmentation_mask, 0.5, 0)
                del segmentation_mask

            label2 = self.create_label_from_image(image_transformed, widget)

        # Add another label beneath the image label for text. The text says "Red: blood vessel, Green: Glomerulus, Blue: Unknown".
        text_label = PyQt5.QtWidgets.QLabel(self.tabbed_interface)
        text_label.setText("Red: blood vessel, Green: Glomerulus, Blue: Unknown")
        text_label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        text_label.show()

        layout.addWidget(label)
        if model_data_manager.dataset_exists(alt_dataset):
            layout.addWidget(label2)
        layout.addWidget(text_label)


        self.tabbed_interface.addTab(widget, clicked_data_entry)

        # Add the tab into the list of tabs.
        self.total_tabs.append(widget)

        # Set the current tab to be the newly created tab.
        self.tabbed_interface.setCurrentWidget(widget)

    def create_label_from_image(self, image_np, parent):
        # Convert the image into a QImage
        height, width, channel = image_np.shape
        bytes_per_line = 3 * width
        q_image = PyQt5.QtGui.QImage(image_np.data, width, height, bytes_per_line, PyQt5.QtGui.QImage.Format_RGB888)

        # Create a QLabel, and set the image as the pixmap of the QLabel.
        # The QLabel is to be contained inside a QVBoxLayout.
        label = PyQt5.QtWidgets.QLabel(parent)
        label.setPixmap(PyQt5.QtGui.QPixmap.fromImage(q_image))
        label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        label.resize(width, height)
        label.show()

        return label

    def custom_algorithm_button_clicked(self):
        current_tab_title = str(self.tabbed_interface.tabText(self.tabbed_interface.currentIndex()))
        print(current_tab_title)

        # Load the image here
        image = cv2.imread(os.path.join(config.input_data_path, "train", current_tab_title + ".tif"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image2 = self.custom_image_transform(image)

        # Display both image and image2 in a new popup window belonging to the main window, with QDialong. The main window is blocked until the popup window is closed.

        popup_window = PyQt5.QtWidgets.QDialog(self)
        popup_window.setWindowTitle("Custom Algorithm Result")
        popup_window.resize(800, 600)

        layout = PyQt5.QtWidgets.QVBoxLayout()
        popup_window.setLayout(layout)

        label = self.create_label_from_image(image, popup_window)
        label2 = self.create_label_from_image(image2, popup_window)

        layout.addWidget(label)
        layout.addWidget(label2)

        popup_window.exec_()

    def custom_image_transform(self, image):
        # This image is a numpy array of shape (height, width, channel) in RGB. The values are uint8. Do k-means on the image, and return the transformed image.
        # The transformed image should be of the same shape and type as the input image.

        # Convert the image into a 2D array of shape (height * width, channel)
        height, width, channel = image.shape
        image_2d = image.reshape((height * width, channel)).astype(np.float64)

        # Do k-means on the image
        kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0, n_init=10).fit(image_2d)

        # Get the cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Get the labels
        labels = kmeans.labels_

        # Get the transformed image
        image_transformed = cluster_centers[labels].reshape((height, width, channel))

        # Clip the values to be between 0 and 255, and convert the image back to uint8
        image_transformed = np.clip(image_transformed, 0, 255).astype(np.uint8)

        return image_transformed


if __name__ == "__main__":
    app = PyQt5.QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()