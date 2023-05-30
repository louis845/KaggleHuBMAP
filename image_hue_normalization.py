import model_data_manager

import argparse

import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Normalize the hues of an image.")

model_data_manager.transform_add_argparse_arguments(parser, requires_model=False)

args = parser.parse_args()

input_data_loader, output_data_writer, _ = model_data_manager.transform_get_argparse_arguments(args, requires_model=False)

for image in model_data_manager.data_information.index:
    image_np = input_data_loader.get_image_data(image)
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(dtype=np.float32) / 255.0

    mean_hue = image_hsv[:, :, 0].mean()

    image_hsv[:, :, 0] = np.clip(image_hsv[:, :, 0] - mean_hue + 0.5, a_min=0.0, a_max=1.0)
    image_hsv = (image_hsv * 255.0).astype(dtype=np.uint8)

    image_np = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    output_data_writer.write_image_data(image, image_np)


input_data_loader.close()
output_data_writer.close()