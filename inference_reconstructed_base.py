# Loads images from the data file, reconstructs a larger image using neighbors.
import os
import config
import cv2
import numpy as np
import pandas as pd
import torch

data_information = pd.read_csv(os.path.join(config.input_data_path, "tile_meta.csv"), index_col=0)

def load_single_image(tile_id, test_or_train="train") -> np.ndarray:
    image = cv2.imread(os.path.join(config.input_data_path, test_or_train, "{}.tif".format(tile_id)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_image(tile_id, image_size=768, test_or_train="train") -> (np.ndarray, np.ndarray):
    image = np.zeros(shape=(image_size, image_size, 3), dtype=np.uint8)
    region_mask = np.zeros(shape=(image_size, image_size), dtype=bool)
    # the 512x512 center of the image is the original image
    """
    image[image_size // 2 - 256:image_size // 2 + 256, image_size // 2 - 256:image_size // 2 + 256, :] = load_single_image(tile_id, test_or_train)
    region_mask[image_size // 2 - 256:image_size // 2 + 256, image_size // 2 - 256:image_size // 2 + 256] = True
    """

    x = data_information.loc[tile_id, "i"]
    y = data_information.loc[tile_id, "j"]

    diff = image_size // 2 - 256

    rel_entries = (data_information["i"] > (x - diff - 512)) & (data_information["i"] < (x + diff + 512))\
                  & (data_information["j"] > (y - diff - 512)) & (data_information["j"] < (y + diff + 512))\
                  & (data_information["source_wsi"] == data_information.loc[tile_id, "source_wsi"])

    for rel_tile_id in data_information.loc[rel_entries].index:
        rel_image = load_single_image(rel_tile_id, test_or_train)
        rel_x = data_information.loc[rel_tile_id, "i"]
        rel_y = data_information.loc[rel_tile_id, "j"]
        rel_width, rel_height = 512, 512

        # crop rel_image if needed
        if rel_x - (x - diff) < 0:
            rel_image = rel_image[:, (x - diff) - rel_x:, :]
            rel_width = rel_image.shape[1]
            rel_x = x - diff
        if rel_x - (x - diff) + rel_width > image_size:
            rel_image = rel_image[:, :image_size - (rel_x - (x - diff) + rel_width), :]
            rel_width = rel_image.shape[1]
        if rel_y - (y - diff) < 0:
            rel_image = rel_image[(y - diff) - rel_y:, :, :]
            rel_height = rel_image.shape[0]
            rel_y = y - diff
        if rel_y - (y - diff) + rel_height > image_size:
            rel_image = rel_image[:image_size - (rel_y - (y - diff) + rel_height), :, :]
            rel_height = rel_image.shape[0]

        # paste rel_image into image
        image[rel_y - (y - diff):rel_y - (y - diff) + rel_height, rel_x - (x - diff):rel_x - (x - diff) + rel_width, :] = rel_image
        region_mask[rel_y - (y - diff):rel_y - (y - diff) + rel_height, rel_x - (x - diff):rel_x - (x - diff) + rel_width] = True

    return image, region_mask

def combine_image_and_region(image, region_mask) -> np.ndarray:
    return np.concatenate([image, np.expand_dims(region_mask, axis=-1)], axis=-1)

def load_combined(tile_id, image_size=768, test_or_train="train") -> torch.Tensor:
    image, region_mask = load_image(tile_id, image_size, test_or_train)
    return torch.tensor(combine_image_and_region(image, region_mask), dtype=torch.float32, device=config.device)\
        .permute(2, 0, 1)

if __name__ == "__main__":
    image, region_mask = load_image("0033bbc76b6b", test_or_train="train")
    cv2.imshow("image", image)
    cv2.waitKey(0)

    cv2.imshow("region_mask", region_mask.astype(np.uint8) * 255)
    cv2.waitKey(0)

    import obtain_re