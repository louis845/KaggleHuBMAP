"""Create a PyQt5 GUI to load a single image "test/72e40acccadf.tif" and display it using a label. There is a second label on the bottom showing the rotated image, where the angle can be adjusted by a slider."""
import math

import PyQt5.QtWidgets
import PyQt5.QtCore
import PyQt5.QtGui
import cv2
import numpy as np
import os
import config

import torch
import torchvision
import torchvision.transforms.functional

class MainWindow(PyQt5.QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image rotation")

        self.resize(1920, 1080)

        self.main_widget = PyQt5.QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.main_layout = PyQt5.QtWidgets.QVBoxLayout(self.main_widget)

        # Create a label to display the image.
        self.image_label = PyQt5.QtWidgets.QLabel(self.main_widget)
        self.main_layout.addWidget(self.image_label)

        # Create a label to display the rotated image.
        self.rotated_image_label = PyQt5.QtWidgets.QLabel(self.main_widget)
        self.main_layout.addWidget(self.rotated_image_label)

        # Create a slider to adjust the rotation angle.
        self.rotation_slider = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Horizontal, self.main_widget)
        self.rotation_slider.setMinimum(0)
        self.rotation_slider.setMaximum(360)
        self.rotation_slider.setValue(0)
        self.main_layout.addWidget(self.rotation_slider)

        # Connect the slider's valueChanged signal to the function below.
        self.rotation_slider.valueChanged.connect(self.slider_value_changed)

        # Load the image.
        self.image = cv2.imread(os.path.join(config.input_data_path, "test", "72e40acccadf.tif"))

        # Set the image label's pixmap to display the image.
        self.image_label.setPixmap(PyQt5.QtGui.QPixmap.fromImage(self.convert_cv2_to_qimage(self.image)))

        # Set the rotated image label's pixmap to display the image.
        self.rotated_image_label.setPixmap(PyQt5.QtGui.QPixmap.fromImage(self.convert_cv2_to_qimage(
            self.rotate_image(self.image, 0)
        )))

    def slider_value_changed(self):
        self.rotated_image_label.setPixmap(PyQt5.QtGui.QPixmap.fromImage(self.convert_cv2_to_qimage(self.rotate_image(self.image, self.rotation_slider.value()))))
        # self.rotate_image(self.image, self.rotation_slider.value())

    def rotate_image(self, image, angle):
        """# Rotate the image by the specified angle.
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, image)"""

        img_tensor = torch.tensor(image, device=config.device, dtype=torch.float32) / 255.0
        img_tensor_rotate = torchvision.transforms.functional.rotate(img_tensor.permute(2, 0, 1), angle).permute(1, 2, 0)

        img_rotate = (img_tensor_rotate.cpu().numpy() * 255.0).astype(np.uint8).copy()

        assert img_rotate.shape == image.shape
        assert np.max(img_rotate) <= 255
        assert np.min(img_rotate) >= 0

        rads = math.radians(angle % 90.0)
        lims = 0.5 / (math.sin(rads) + math.cos(rads))

        # Draw rectangle with edges (centerx - imagewidth * lims, centery - imageheight * lims) to (centerx + imagewidth * lims, centery + imageheight * lims)
        cv2.rectangle(img_rotate,
                      (int(img_rotate.shape[1] * (0.5 - lims)), int(img_rotate.shape[0] * (0.5 - lims))), (int(img_rotate.shape[1] * (0.5 + lims)), int(img_rotate.shape[0] * (0.5 + lims))), (255, 255, 255), 3)

        return img_rotate

    def convert_cv2_to_qimage(self, image):
        # Convert the image from OpenCV to PyQt.
        image = image.copy()
        height, width, colors = image.shape
        bytes_per_line = 3 * width
        return PyQt5.QtGui.QImage(image.data, width, height, bytes_per_line, PyQt5.QtGui.QImage.Format_RGB888)

if __name__ == "__main__":
    app = PyQt5.QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()