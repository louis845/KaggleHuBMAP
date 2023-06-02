import os
import gc

import cv2
import pandas as pd
import numpy as np
import torch
import time

import config
import sklearn

data_information = pd.read_csv(os.path.join(config.input_data_path, "tile_meta.csv"), index_col=0)

def compute_image_info(path_to_img):
    """
    Compute the image info for a given image.
    :param path_to_img: Path to the image.
    :return: A dictionary containing the image info.
    """
    # Load the image
    img = cv2.imread(path_to_img)
    # Convert the image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_hsv_torch = torch.tensor(img_hsv, dtype=torch.uint8, device=config.device)
    # this is a width x height x 3 tensor, with integer values between 0 and 255. For each channel, create a length 255 torch array with the number of pixels with that value.
    # For example, if there are 10 pixels with value 0, then the first element of the array is 10.

    with torch.no_grad():
        reference = torch.arange(0, 256, dtype=torch.uint8, device=config.device)
        hue_distribution = torch.sum(img_hsv_torch[:, :, 0].unsqueeze(2) == reference.unsqueeze(0).unsqueeze(0), dim=(0, 1))
        saturation_distribution = torch.sum(img_hsv_torch[:, :, 1].unsqueeze(2) == reference.unsqueeze(0).unsqueeze(0), dim=(0, 1))
        value_distribution = torch.sum(img_hsv_torch[:, :, 2].unsqueeze(2) == reference.unsqueeze(0).unsqueeze(0), dim=(0, 1))
    return {"hue": hue_distribution, "saturation": saturation_distribution, "value": value_distribution, "pixels": int(img.shape[0]) * int(img.shape[1])}

if __name__ == "__main__":
    if not os.path.exists("hsv_distributions"):
        os.mkdir("hsv_distributions")


    # Loop through each unique WSI, and for each WSI, loop through each unique dataset. Compute the image info for each dataset using the compute_image_info function.
    unique_wsi = data_information["source_wsi"].unique()
    for wsi in unique_wsi:
        image_group = data_information[data_information["source_wsi"] == wsi]
        print("Computing image info for WSI {} with {} images".format(wsi, len(image_group)))
        ctime = time.time()
        hue = torch.zeros(size=(256,), dtype=torch.float32, device=config.device)
        saturation = torch.zeros(size=(256,), dtype=torch.float32, device=config.device)
        value = torch.zeros(size=(256,), dtype=torch.float32, device=config.device)
        pixels = 0
        for image in image_group.index:
            info = compute_image_info(os.path.join(config.input_data_path, "train", image + ".tif"))
            with torch.no_grad():
                hue += info["hue"]
                saturation += info["saturation"]
                value += info["value"]
                pixels += info["pixels"]
            del info
            gc.collect()

        with torch.no_grad():
            hue = hue / pixels
            saturation = saturation / pixels
            value = value / pixels
        hue = hue.cpu().numpy()
        saturation = saturation.cpu().numpy()
        value = value.cpu().numpy()
        np.savez(os.path.join("hsv_distributions", "wsi_{}.npz".format(wsi)), hue=hue, saturation=saturation, value=value)

        print("Computed image info for WSI {} in {} seconds".format(wsi, time.time() - ctime))