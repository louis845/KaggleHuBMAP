import model_data_manager

import argparse

import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Copies the image.")

model_data_manager.transform_add_argparse_arguments(parser, requires_model=False)

args = parser.parse_args()

input_data_loader, output_data_writer, _ = model_data_manager.transform_get_argparse_arguments(args, requires_model=False)

for image in model_data_manager.data_information.index:
    image_np = input_data_loader.get_image_data(image)
    output_data_writer.write_image_data(image, image_np)


input_data_loader.close()
output_data_writer.close()