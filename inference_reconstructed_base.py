# Loads images from the data file, reconstructs a larger image using neighbors.
import os
import config
import cv2
import numpy as np
import pandas as pd

data_information = pd.read_csv(os.path.join(config.input_data_path, "tile_meta.csv"))

def load_single_image(tile_id, test_or_train="train"):
    image = cv2.imread(os.path.join(config.input_data_path, test_or_train, "{}.png".format(tile_id)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_image(tile_id, image_size=768, test_or_train="train"):
    image = np.zeros(shape=(image_size, image_size, 3), dtype=np.uint8)
    # the 512x512 center of the image is the original image
    image[image_size // 2 - 256:image_size // 2 + 256, image_size // 2 - 256:image_size // 2 + 256, :] = load_single_image(tile_id, test_or_train)

    data_information