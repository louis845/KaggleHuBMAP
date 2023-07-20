"""Python file to visualize the data. This program contains a main GUI window, with Qt PyQt5."""


import os
import json
import traceback
import PyQt5
import PyQt5.QtWidgets
import PyQt5.QtCore

import pandas as pd
import numpy as np
import torch
import torchvision.transforms.functional
import h5py

import cv2
import sklearn.cluster

import config
import inference_reconstructed_base

import model_data_manager

with open(os.path.join(config.input_data_path, "polygons.jsonl")) as json_file:
    json_list = list(json_file)

all_polygon_masks = {}
for json_str in json_list:
    polygon_masks = json.loads(json_str)
    all_polygon_masks[polygon_masks["id"]] = polygon_masks["annotations"]

segmentation_store = h5py.File(os.path.join("segmentation_data", "data_summary.h5"), "r")


instance_segmentation_previous_values = {} # type: dict[str, any]
# stores the previous values for the instance segmentation popup
class InstanceSegmentationPopup(PyQt5.QtWidgets.QDialog):
    """Opens up a popup window to display instance segmentation tasks.
    The constructor accepts two numpy arrays of shape (H, W, 3) as inputs, the first array is uint8 the original image,
    the second array should be float32 denoting the class probabilities. The GUI should contain the following:

    At the top center, there should be a widget, and inside the widget two labels horizontally, one displaying the
    original image, and one displaying the processed predictions from the class probabilities.

    Below that, there should be options for choosing the parameters for the processing step. The options should be
    displayed vertically, and each option should have a label on the left for the name of the option, and a widget on
    the right for setting the option. The options are listed below.

    Option 1: Processing mode. This should be a dropdown menu which uses the list PROCESSING_MODES
    Option 2: Background threshold. This should be a slider with a range of 0 to 1, with a default value of 0.5
    Option 3: Boundary threshold. This should be a slider with a range of 0 to 1, with a default value of 0.5
    Option 4: Boundary erosion. This should be a slider with a range of 0 to 10, with a default value of 0, increment 1
    Option 5: Instances. This should be a checkbox, default unchecked.

    At the bottom, there should be a button called "Compute", which when pressed, should compute the instances with the processing parameters
    """
    PROCESSING_INTERIOR_ONLY = "INTERIOR_ONLY"
    PROCESSING_BOUNDARY_ONLY = "BOUNDARY_ONLY"
    PROCESSING_INTERIOR_AND_BOUNDARY = "INTERIOR_AND_BOUNDARY"
    PROCESSING_MODES = [PROCESSING_INTERIOR_ONLY, PROCESSING_BOUNDARY_ONLY, PROCESSING_INTERIOR_AND_BOUNDARY]

    def __init__(self, original_image: np.ndarray, class_probabilities: np.ndarray, parent=None):
        super().__init__(parent)
        self.original_image = original_image
        self.class_probabilities = class_probabilities
        assert self.original_image.shape == self.class_probabilities.shape
        assert self.original_image.dtype == np.uint8
        assert self.class_probabilities.dtype == np.float32

        self.setWindowTitle("Instance Segmentation")
        self.resize(1920, 1080)
        self.setup_ui()
        self.load_previous_values()

    def setup_ui(self):
        # create main widget and QVBoxLayout
        self.main_widget = PyQt5.QtWidgets.QWidget(self)
        self.main_layout = PyQt5.QtWidgets.QVBoxLayout(self.main_widget)

        # create top widget and QHBoxLayout
        self.top_widget = PyQt5.QtWidgets.QWidget(self.main_widget)
        self.top_layout = PyQt5.QtWidgets.QHBoxLayout(self.top_widget)

        # create original image label and add to top layout
        self.original_image_label = PyQt5.QtWidgets.QLabel(self.top_widget)
        self.original_image_label.setPixmap(PyQt5.QtGui.QPixmap.fromImage(PyQt5.QtGui.QImage(self.original_image.data,
                                            self.original_image.shape[1], self.original_image.shape[0], self.original_image.shape[1] * 3, PyQt5.QtGui.QImage.Format_RGB888)))
        self.top_layout.addWidget(self.original_image_label)

        # create processed image label and add to top layout, but don't set the image yet. Set to default black image with same size
        self.processed_image_label = PyQt5.QtWidgets.QLabel(self.top_widget)
        self.processed_image_label.setPixmap(PyQt5.QtGui.QPixmap.fromImage(PyQt5.QtGui.QImage(np.zeros_like(self.original_image).data,
                                            self.original_image.shape[1], self.original_image.shape[0], self.original_image.shape[1] * 3, PyQt5.QtGui.QImage.Format_RGB888)))
        self.top_layout.addWidget(self.processed_image_label)

        # add top widget to main layout
        self.main_layout.addWidget(self.top_widget)

        # create processing mode widget and add to main layout
        self.processing_mode_widget = PyQt5.QtWidgets.QWidget(self.main_widget)
        self.processing_mode_layout = PyQt5.QtWidgets.QHBoxLayout(self.processing_mode_widget)
        self.processing_mode_label = PyQt5.QtWidgets.QLabel(self.processing_mode_widget)
        self.processing_mode_label.setText("Processing Mode")
        self.processing_mode_layout.addWidget(self.processing_mode_label)
        self.processing_mode_dropdown = PyQt5.QtWidgets.QComboBox(self.processing_mode_widget)
        self.processing_mode_dropdown.addItems(self.PROCESSING_MODES)
        self.processing_mode_layout.addWidget(self.processing_mode_dropdown)
        self.main_layout.addWidget(self.processing_mode_widget)

        # create background threshold widget and add to main layout
        self.background_threshold_widget = PyQt5.QtWidgets.QWidget(self.main_widget)
        self.background_threshold_layout = PyQt5.QtWidgets.QHBoxLayout(self.background_threshold_widget)
        self.background_threshold_label = PyQt5.QtWidgets.QLabel(self.background_threshold_widget)
        self.background_threshold_label.setText("Background Threshold")
        self.background_threshold_layout.addWidget(self.background_threshold_label)
        self.background_threshold_slider = PyQt5.QtWidgets.QSlider(self.background_threshold_widget)
        self.background_threshold_slider.setOrientation(PyQt5.QtCore.Qt.Horizontal)
        self.background_threshold_slider.setRange(0, 100)
        self.background_threshold_slider.setValue(50)
        self.background_threshold_layout.addWidget(self.background_threshold_slider)
        self.main_layout.addWidget(self.background_threshold_widget)

        # create boundary threshold widget and add to main layout
        self.boundary_threshold_widget = PyQt5.QtWidgets.QWidget(self.main_widget)
        self.boundary_threshold_layout = PyQt5.QtWidgets.QHBoxLayout(self.boundary_threshold_widget)
        self.boundary_threshold_label = PyQt5.QtWidgets.QLabel(self.boundary_threshold_widget)
        self.boundary_threshold_label.setText("Boundary Threshold")
        self.boundary_threshold_layout.addWidget(self.boundary_threshold_label)
        self.boundary_threshold_slider = PyQt5.QtWidgets.QSlider(self.boundary_threshold_widget)
        self.boundary_threshold_slider.setOrientation(PyQt5.QtCore.Qt.Horizontal)
        self.boundary_threshold_slider.setRange(0, 100)
        self.boundary_threshold_slider.setValue(50)
        self.boundary_threshold_layout.addWidget(self.boundary_threshold_slider)
        self.main_layout.addWidget(self.boundary_threshold_widget)

        # create boundary erosion widget and add to main layout
        self.boundary_erosion_widget = PyQt5.QtWidgets.QWidget(self.main_widget)
        self.boundary_erosion_layout = PyQt5.QtWidgets.QHBoxLayout(self.boundary_erosion_widget)
        self.boundary_erosion_label = PyQt5.QtWidgets.QLabel(self.boundary_erosion_widget)
        self.boundary_erosion_label.setText("Boundary Erosion")
        self.boundary_erosion_layout.addWidget(self.boundary_erosion_label)
        self.boundary_erosion_slider = PyQt5.QtWidgets.QSlider(self.boundary_erosion_widget)
        self.boundary_erosion_slider.setOrientation(PyQt5.QtCore.Qt.Horizontal)
        self.boundary_erosion_slider.setRange(0, 10)
        self.boundary_erosion_slider.setValue(0)
        self.boundary_erosion_layout.addWidget(self.boundary_erosion_slider)
        self.main_layout.addWidget(self.boundary_erosion_widget)

        # create instances widget and add to main layout
        self.instances_widget = PyQt5.QtWidgets.QWidget(self.main_widget)
        self.instances_layout = PyQt5.QtWidgets.QHBoxLayout(self.instances_widget)
        self.instances_label = PyQt5.QtWidgets.QLabel(self.instances_widget)
        self.instances_label.setText("Instances")
        self.instances_layout.addWidget(self.instances_label)
        self.instances_checkbox = PyQt5.QtWidgets.QCheckBox(self.instances_widget)
        self.instances_checkbox.setChecked(False)
        self.instances_layout.addWidget(self.instances_checkbox)
        self.main_layout.addWidget(self.instances_widget)


        # add compute button to main layout
        self.compute_button = PyQt5.QtWidgets.QPushButton(self.main_widget)
        self.compute_button.setText("Compute")
        self.compute_button.clicked.connect(self.compute_button_clicked)
        self.main_layout.addWidget(self.compute_button)

        # add main widget to main window
        self.setCentralWidget(self.main_widget)

    def load_previous_values(self):
        global instance_segmentation_previous_values
        if "processing_mode" in instance_segmentation_previous_values:
            self.processing_mode_dropdown.setCurrentIndex(self.PROCESSING_MODES.index(instance_segmentation_previous_values["processing_mode"]))
        if "background_threshold" in instance_segmentation_previous_values:
            self.background_threshold_slider.setValue(instance_segmentation_previous_values["background_threshold"])
        if "boundary_threshold" in instance_segmentation_previous_values:
            self.boundary_threshold_slider.setValue(instance_segmentation_previous_values["boundary_threshold"])
        if "boundary_erosion" in instance_segmentation_previous_values:
            self.boundary_erosion_slider.setValue(instance_segmentation_previous_values["boundary_erosion"])
        if "instances" in instance_segmentation_previous_values:
            self.instances_checkbox.setChecked(instance_segmentation_previous_values["instances"])

    def closeEvent(self, event):
        global instance_segmentation_previous_values
        instance_segmentation_previous_values["processing_mode"] = self.PROCESSING_MODES[self.processing_mode_dropdown.currentIndex()]
        instance_segmentation_previous_values["background_threshold"] = self.background_threshold_slider.value()
        instance_segmentation_previous_values["boundary_threshold"] = self.boundary_threshold_slider.value()
        instance_segmentation_previous_values["boundary_erosion"] = self.boundary_erosion_slider.value()
        instance_segmentation_previous_values["instances"] = self.instances_checkbox.isChecked()
        self.closed.emit()
        event.accept()

    def compute_button_clicked(self):
        # get processing mode
        processing_mode = self.PROCESSING_MODES[self.processing_mode_dropdown.currentIndex()]
        background_threshold = self.background_threshold_slider.value() / 100
        boundary_threshold = self.boundary_threshold_slider.value() / 100
        boundary_erosion = self.boundary_erosion_slider.value()
        compute_instances = self.instances_checkbox.isChecked()

        class_probas_torch = torch.from_numpy(self.class_probas).to(config.device)
        # compute processed image
        if processing_mode == self.PROCESSING_INTERIOR_ONLY:
            mask = inference_reconstructed_base.get_objectness_mask(class_probas_torch, background_threshold)
        elif processing_mode == self.PROCESSING_BOUNDARY_ONLY:
            mask = inference_reconstructed_base.get_boundary_mask(class_probas_torch, boundary_threshold, boundary_erosion)
        elif processing_mode == self.PROCESSING_INTERIOR_AND_BOUNDARY:
            mask = inference_reconstructed_base.get_instance_mask(class_probas_torch, boundary_threshold, background_threshold, boundary_erosion)

        image = inference_reconstructed_base.get_instances_image(mask, instances=compute_instances) # (H, W, 3) numpy RGB image

        # set processed image label
        self.processed_image_label.setPixmap(PyQt5.QtGui.QPixmap.fromImage(PyQt5.QtGui.QImage(image.data,
                                    image.shape[1], image.shape[0], image.shape[1] * 3, PyQt5.QtGui.QImage.Format_RGB888)))


def load_tile_with_polygons(tile_id:str):
    # Load the image here
    image = cv2.imread(os.path.join(config.input_data_path, "train", tile_id + ".tif"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Add the polygon annotations if it exists
    if tile_id in all_polygon_masks:
        for polygon_mask in all_polygon_masks[tile_id]:
            # The color depends on the type, default unknown color = blue
            color = (0, 0, 255)
            if polygon_mask["type"] == "glomerulus":
                color = (0, 255, 0)  # green
            elif polygon_mask["type"] == "blood_vessel":
                color = (255, 0, 0)  # red

            # Draw the polygon
            polygon_coordinate_list = polygon_mask["coordinates"][
                0]  # This is a list of integer 2-tuples, representing the coordinates.
            for i in range(len(polygon_coordinate_list)):
                cv2.line(image, polygon_coordinate_list[i],
                         polygon_coordinate_list[(i + 1) % len(polygon_coordinate_list)], color, 3)

            # Fill the polygon with the color, with 35% opacity
            overlay = image.copy()
            cv2.fillPoly(overlay, [np.array(polygon_coordinate_list)], color)
            image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    return image

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
        self.checkboxes_widget.setGeometry(0, 100, 150, 980)
        self.buttons_scroll_area = PyQt5.QtWidgets.QScrollArea(self.left_selection_area)
        self.buttons_scroll_area.setGeometry(150, 100, 150, 980)
        self.subdata_restriction = PyQt5.QtWidgets.QComboBox(self.left_selection_area)
        self.subdata_restriction.setGeometry(0, 0, 300, 100)
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
        for segmentation in segmentation_store["segmentation_data"][model_data_manager.data_information.index[0]].keys():
            self.comparison_segmentation_dropdown.addItem("Segmentation data: " + segmentation)

        # Loop through the existing subdata, set the options for the subdata dropdown.
        self.subdata_restriction.addItem("None")
        for subdata in model_data_manager.list_subdata():
            self.subdata_restriction.addItem(subdata)

        self.subdata_restriction.currentIndexChanged.connect(self.checkbox_clicked)


    def checkbox_clicked(self):
        # Loop through all the buttons, and hide/show the buttons depending on which checkboxes are checked.
        subdata_chosen = str(self.subdata_restriction.currentText())
        if subdata_chosen != "None":
            subdata_list = model_data_manager.get_subdata_entry_list(subdata_chosen)

        for button in self.buttons:
            button_dataset = model_data_manager.data_information.loc[button.text(), "dataset"]
            visible = self.dataset_checkboxes[button_dataset].isChecked() and self.wsi_checkboxes[model_data_manager.data_information.loc[button.text(), "source_wsi"]].isChecked()
            if subdata_chosen != "None":
                visible = visible and (str(button.text()) in subdata_list)

            button.setVisible(visible)

    def selection_button_clicked(self, clicked_data_entry):
        try:
            # Create a new tab in the tabbed interface, and add the data visualization for the selected dataset into the new tab.
            # Load an image from os.path.join(config.input_data_path, "train", clicked_data_entry + ".tif") using cv2, and save it as a numpy array.
            # The image is to be displayed in a Qt widget, so it needs to be converted into a QImage first.

            # Load the image here
            image = load_tile_with_polygons(clicked_data_entry)

            widget = PyQt5.QtWidgets.QWidget()
            layout = PyQt5.QtWidgets.QVBoxLayout()
            widget.setLayout(layout)

            label = self.create_label_from_image(image, widget)
            alt_dataset = str(self.comparison_dropdown.currentText())
            segmentation_dataset = str(self.comparison_segmentation_dropdown.currentText())

            if model_data_manager.dataset_exists(alt_dataset):
                data_loader = model_data_manager.get_dataset_dataloader(alt_dataset)
                image_transformed = np.array(data_loader.get_image_data(clicked_data_entry)).astype(dtype=np.uint8)

                if image_transformed.shape[2] == 1:
                    image_transformed = np.repeat(image_transformed, 3, axis=2)
                    image_transformed = image_transformed.astype(dtype=np.uint8)

                data_loader.close()
                del data_loader

                # Add the segmentation mask as a white overlay if it exists
                if segmentation_dataset != "None":
                    if segmentation_dataset.startswith("Segmentation data: "):
                        try:
                            dataset_name = segmentation_dataset[len("Segmentation data: "):]
                            segmentation_mask = np.array(segmentation_store["segmentation_data"][clicked_data_entry][dataset_name], dtype=np.uint8) * 255

                            if len(segmentation_mask.shape) == 2:
                                segmentation_mask = np.repeat(np.expand_dims(segmentation_mask, axis=2), axis=2, repeats=3)

                            if segmentation_mask.shape[2] == 1:
                                segmentation_mask = np.repeat(segmentation_mask, 3, axis=2)

                            image_transformed = cv2.addWeighted(image_transformed, 0.5, segmentation_mask, 0.5, 0)
                            del segmentation_mask
                        except Exception as e:
                            traceback.print_exc()
                    elif model_data_manager.dataset_exists(segmentation_dataset):
                        data_loader = model_data_manager.get_dataset_dataloader(segmentation_dataset)
                        segmentation_mask = np.array(data_loader.get_image_data(clicked_data_entry)).astype(dtype=np.uint8)
                        data_loader.close()
                        del data_loader

                        if len(segmentation_mask.shape) == 2:
                            segmentation_mask = np.repeat(np.expand_dims(segmentation_mask, axis=2), axis=2, repeats=3)

                        if segmentation_mask.shape[2] == 1:
                            segmentation_mask = np.repeat(segmentation_mask, 3, axis=2)

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
        except Exception as e:
            traceback.print_exc()

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
        segmentation_dataset = str(self.comparison_segmentation_dropdown.currentText())
        if segmentation_dataset == "None":
            dialog = PyQt5.QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Error")
            dialog.setText("You must select a segmentation dataset containing the probabilities")
            dialog.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            dialog.exec_()
            return
        if not model_data_manager.dataset_exists(segmentation_dataset):
            dialog = PyQt5.QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Error")
            dialog.setText("The segmentation dataset you selected does not exist")
            dialog.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            dialog.exec_()
            return

        current_tab_title = str(self.tabbed_interface.tabText(self.tabbed_interface.currentIndex()))
        # Load the image here
        image = load_tile_with_polygons(current_tab_title)
        data_loader = model_data_manager.get_dataset_dataloader(segmentation_dataset)
        probas = np.array(data_loader.get_image_data(current_tab_title))
        data_loader.close()
        del data_loader

        # create instance segmentation popup
        popup = InstanceSegmentationPopup(image, probas, self)
        popup.exec_()

    def apply_random_shear(self, image, displacement_field, xory="x"):
        x = np.random.randint(low=0, high=image.shape[2])
        y = np.random.randint(low=0, high=image.shape[1])
        sigma = np.random.uniform(low=100.0, high=200.0)
        magnitude = np.random.uniform(low=10000.0, high=16000.0) * np.random.choice([-1, 1])

        width = 384

        expand_left = min(x, width)
        expand_right = min(image.shape[2] - x, width + 1)
        expand_top = min(y, width)
        expand_bottom = min(image.shape[1] - y, width + 1)

        if xory == "x":
            displacement_field[0, x - expand_left:x + expand_right, y - expand_top:y + expand_bottom, 0:1] += \
                (np.expand_dims(cv2.getGaussianKernel(ksize=width * 2 + 1, sigma=sigma), axis=-1) * cv2.getGaussianKernel(
                    ksize=width * 2 + 1, sigma=sigma) * magnitude)[width - expand_left:width + expand_right,
                width - expand_top:width + expand_bottom, :]
        else:
            displacement_field[0, x - expand_left:x + expand_right, y - expand_top:y + expand_bottom, 1:2] += \
                (np.expand_dims(cv2.getGaussianKernel(ksize=width * 2 + 1, sigma=sigma),
                                axis=-1) * cv2.getGaussianKernel(
                    ksize=width * 2 + 1, sigma=sigma) * magnitude)[width - expand_left:width + expand_right,
                width - expand_top:width + expand_bottom, :]

    def custom_image_transform(self, image):
        try:
            color = (255, 255, 255)
            thickness = 3
            grid_size = 64
            height, width, channels = image.shape
            for x in range(0, width, grid_size):
                cv2.line(image, (x, 0), (x, height), color, thickness)
            for y in range(0, height, grid_size):
                cv2.line(image, (0, y), (width, y), color, thickness)


            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            """displacement_field = torch.randn(size=(2, image.shape[1], image.shape[2]), dtype=torch.float32)
            displacement_field = torchvision.transforms.functional.gaussian_blur(displacement_field, kernel_size=101, sigma=100.0).permute(1, 2, 0).unsqueeze(0) """
            displacement_field = np.zeros(shape=(1, image.shape[1], image.shape[2], 2), dtype=np.float32)

            for k in range(4):
                self.apply_random_shear(image, displacement_field, xory="x")
                self.apply_random_shear(image, displacement_field, xory="y")

            displacement_field = torch.tensor(displacement_field, dtype=torch.float32)
            image = torchvision.transforms.functional.elastic_transform(image, displacement_field).permute(1, 2, 0)
            image = image.numpy().astype(dtype=np.uint8).copy()
            return image
        except Exception as e:
            traceback.print_exc()
        return image
        """try:
            # This image is a numpy array of shape (height, width, channel) RGB image. Do hierarchical clustering with BisectingKMeans on the image into n clusters, and return the image with the clusters colored.
            n_clusters = 16

            height, width, channel = image.shape
            image = np.reshape(image, (height * width, channel)).astype(np.float64)

            # Do clustering. cluster_array is a deep nested list representing the cluster tree, where each list has either one or two elements.
            cluster_tree = {"items": np.ones((height * width), dtype=bool), "center":np.mean(image, axis=0)}

            for i in range(n_clusters - 1):
                # Loop through the leaves of the cluster tree, and find the cluster with the largest SSE.
                # Split the cluster with the largest SSE into two clusters, and add the two clusters into the cluster tree. Use sklearn's KMeans with k=2 to split the cluster.
                # Repeat until there are n clusters in the cluster tree.

                # Find the cluster with the largest SSE.
                max_sse = -1
                max_sse_stack = []

                # Do preorder traversal on cluster_tree, starting from the root node
                stack = []

                while True:
                    current_node = cluster_tree
                    for i in stack:
                        current_node = current_node["subgroups"][i]

                    # Check if the current node is a leaf node
                    if "subgroups" not in current_node:
                        sse = np.mean(np.square(image[current_node["items"], :] - current_node["center"]))

                        if sse > max_sse:
                            max_sse = sse
                            del max_sse_stack
                            max_sse_stack = stack.copy()

                        if len(stack) > 0 and stack[-1] == 0:
                            stack[-1] = 1
                        else:
                            while len(stack) > 0 and stack[-1] == 1:
                                stack.pop()
                            if len(stack) == 0:
                                break
                            stack[-1] = 1
                    else:
                        stack.append(0)

                cluster_to_split = cluster_tree
                for i in max_sse_stack:
                    cluster_to_split = cluster_to_split["subgroups"][i]

                # Split the cluster with the largest SSE into two clusters, and add the two clusters into the cluster tree. Use sklearn's KMeans with k=2 to split the cluster.
                image_cluster_colors = image[cluster_to_split["items"], :]
                cluster_mask = cluster_to_split["items"]

                kmeans = sklearn.cluster.KMeans(n_clusters=2, n_init=10)
                kmeans.fit(image_cluster_colors)

                first_cluster_items = np.zeros_like(cluster_mask, dtype=bool)
                second_cluster_items = np.zeros_like(cluster_mask, dtype=bool)

                first_cluster_items[cluster_mask] = kmeans.labels_ == 0
                second_cluster_items[cluster_mask] = kmeans.labels_ == 1
                cluster_to_split["subgroups"] = [{"items": first_cluster_items, "center": kmeans.cluster_centers_[0]}, {"items": second_cluster_items, "center": kmeans.cluster_centers_[1]}]

            cluster_stacks = np.empty(shape=n_clusters, dtype=object)
            cluster_colors = np.zeros(shape=(n_clusters, 1, 3), dtype=np.float64)
            count = 0

            stack = []
            while True:
                current_node = cluster_tree
                for i in stack:
                    current_node = current_node["subgroups"][i]

                # Check if the current node is a leaf node
                if "subgroups" not in current_node:
                    cluster_stacks[count] = stack.copy()
                    cluster_colors[count, 0, :] = current_node["center"]
                    count += 1

                    if len(stack) > 0 and stack[-1] == 0:
                        stack[-1] = 1
                    else:
                        while len(stack) > 0 and stack[-1] == 1:
                            stack.pop()
                        if len(stack) == 0:
                            break
                        stack[-1] = 1
                else:
                    stack.append(0)

            cluster_colors = cv2.cvtColor(cluster_colors.astype(dtype=np.uint8), cv2.COLOR_RGB2HLS)[:, 0, 1]
            rank = np.argsort(np.argsort(cluster_colors))

            # Assign each pixel to a cluster
            image_clustered = np.zeros_like(image)

            # Do preorder traversal on cluster_tree, starting from the root node
            for k in range(cluster_stacks.shape[0]):
                current_node = cluster_tree
                for i in cluster_stacks[k]:
                    current_node = current_node["subgroups"][i]
                image_clustered[current_node["items"], :] = 255.0 * (rank[k] / (n_clusters - 1))

            image_clustered = np.reshape(image_clustered, (height, width, channel))

            # Clip the image to be between 0 and 255, and convert to uint8
            image_clustered = np.clip(image_clustered, 0, 255).astype(np.uint8)
        except Exception as e:
            # Print in details the error message e and the stack trace
            traceback.print_exc()

            image_clustered = image

        return image_clustered"""


if __name__ == "__main__":
    app = PyQt5.QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()