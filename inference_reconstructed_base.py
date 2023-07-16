# Loads images from the data file, reconstructs a larger image using neighbors.
import os

import h5py

import config
import cv2
import numpy as np
import pandas as pd
import torch

data_information = pd.read_csv(os.path.join(config.input_data_path, "tile_meta.csv"), index_col=0)

def load_single_image(tile_id) -> np.ndarray:
    if os.path.isfile(os.path.join(config.input_data_path, "train", "{}.tif".format(tile_id))):
        test_or_train = "train"
    elif os.path.isfile(os.path.join(config.input_data_path, "test", "{}.tif".format(tile_id))):
        test_or_train = "test"
    else:
        raise ValueError("tile_id {} not found".format(tile_id))

    image = cv2.imread(os.path.join(config.input_data_path, test_or_train, "{}.tif".format(tile_id)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_image(tile_id, image_size=768) -> (np.ndarray, np.ndarray):
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
        rel_image = load_single_image(rel_tile_id)
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

def load_combined(tile_id, image_size=768) -> torch.Tensor:
    image, region_mask = load_image(tile_id, image_size)
    return torch.tensor(combine_image_and_region(image, region_mask), dtype=torch.float32, device=config.device)\
        .permute(2, 0, 1)

# for the 1024 to 512 image inference. stores the 512x512 region that is the center of a 768x768 subregion of the 1024x1024 image.
static_768_region_cache = None
def initialize_region_cache():
    global static_768_region_cache
    if static_768_region_cache is None:
        static_768_region_cache = torch.zeros(size=(512, 512), dtype=torch.bool, device=config.device)

def full_array(shape):
    assert len(shape) == 2, "shape must be 2-dimensional"
    arr = np.empty(shape=shape, dtype="object")
    for i in range(shape[0]):
        for j in range(shape[1]):
            arr[i, j] = []
    return arr

class Composite1024To512ImageInference:
    """Helper class to input a 1024x1024 image, and do inference on the center 512x512 region."""

    CENTER, CENTER_STR = 0, "center"
    TOP_LEFT, TOP_LEFT_STR = 1, "top_left"
    TOP_RIGHT, TOP_RIGHT_STR = 2, "top_right"
    BOTTOM_LEFT, BOTTOM_LEFT_STR = 3, "bottom_left"
    BOTTOM_RIGHT, BOTTOM_RIGHT_STR = 4, "bottom_right"

    LOCATIONS = [CENTER, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT]

    def __init__(self):
        self.image_loaded = False
        self.image = None
        self.region_mask = None
        self.tile_id = None

        self.logits_obtained = False
        self.logits = {}

    def load_image(self, tile_id: str) -> None:
        self.image, self.region_mask = load_image(tile_id, image_size=1024)
        self.image_loaded = True
        self.tile_id = tile_id
        initialize_region_cache()

    def get_combined_image(self, location: int):
        if not self.image_loaded:
            raise ValueError("Image not loaded. Load image first!")
        assert location in [self.CENTER, self.TOP_LEFT, self.TOP_RIGHT, self.BOTTOM_LEFT, self.BOTTOM_RIGHT],\
            "location must be one of the following: Composite1024To512ImageInference.CENTER, Composite1024To512ImageInference.TOP_LEFT, Composite1024To512ImageInference.TOP_RIGHT, Composite1024To512ImageInference.BOTTOM_LEFT, Composite1024To512ImageInference.BOTTOM_RIGHT"

        # self.image is a 1024x1024 image, and return a 768x768 subimage
        if location == self.CENTER:
            return torch.tensor(np.concatenate([self.image[128:-128, 128:-128, :], np.expand_dims(self.region_mask[128:-128, 128:-128], axis=-1)], axis=-1),
                                dtype=torch.float32, device=config.device).permute(2, 0, 1)
        elif location == self.TOP_LEFT:
            return torch.tensor(np.concatenate([self.image[:768, :768, :], np.expand_dims(self.region_mask[:768, :768], axis=-1)], axis=-1),
                                dtype=torch.float32, device=config.device).permute(2, 0, 1)
        elif location == self.TOP_RIGHT:
            return torch.tensor(np.concatenate([self.image[:768, -768:, :], np.expand_dims(self.region_mask[:768, -768:], axis=-1)], axis=-1),
                                dtype=torch.float32, device=config.device).permute(2, 0, 1)
        elif location == self.BOTTOM_LEFT:
            return torch.tensor(np.concatenate([self.image[-768:, :768, :], np.expand_dims(self.region_mask[-768:, :768], axis=-1)], axis=-1),
                                dtype=torch.float32, device=config.device).permute(2, 0, 1)
        elif location == self.BOTTOM_RIGHT:
            return torch.tensor(np.concatenate([self.image[-768:, -768:, :], np.expand_dims(self.region_mask[-768:, -768:], axis=-1)], axis=-1),
                                dtype=torch.float32, device=config.device).permute(2, 0, 1)

    def has_region_information(self, location: int):
        if not self.image_loaded:
            raise ValueError("Image not loaded. Load image first!")
        assert location in [self.CENTER, self.TOP_LEFT, self.TOP_RIGHT, self.BOTTOM_LEFT, self.BOTTOM_RIGHT], \
            "location must be one of the following: Composite1024To512ImageInference.CENTER, Composite1024To512ImageInference.TOP_LEFT, Composite1024To512ImageInference.TOP_RIGHT, Composite1024To512ImageInference.BOTTOM_LEFT, Composite1024To512ImageInference.BOTTOM_RIGHT"

        if location == self.CENTER:
            static_768_region_cache.copy_(torch.from_numpy(self.region_mask[256:-256, 256:-256]))
        elif location == self.TOP_LEFT:
            static_768_region_cache.copy_(torch.from_numpy(self.region_mask[128:640, 128:640]))
        elif location == self.TOP_RIGHT:
            static_768_region_cache.copy_(torch.from_numpy(self.region_mask[128:640, -640:-128]))
        elif location == self.BOTTOM_LEFT:
            static_768_region_cache.copy_(torch.from_numpy(self.region_mask[-640:-128, 128:640]))
        elif location == self.BOTTOM_RIGHT:
            static_768_region_cache.copy_(torch.from_numpy(self.region_mask[-640:-128, -640:-128]))
        has_info = static_768_region_cache.all()
        if (not has_info) and location == self.CENTER:
            print("POTENTIAL ERROR: Center region for tile {} does not have information!".format(self.tile_id))
        return has_info

    def store_prediction_logits(self, location: int, logits: torch.Tensor):
        """
        Store the logits for a 512x512 subregion of the 1024x1024 image, in the location.
        :param location: one of the static variables of Composite1024To512ImageInference
        :param logits: a torch tensor of shape (512, 512, 3, k)
        """
        if not self.image_loaded:
            raise ValueError("Image not loaded. Load image first!")
        assert location in [self.CENTER, self.TOP_LEFT, self.TOP_RIGHT, self.BOTTOM_LEFT, self.BOTTOM_RIGHT], \
            "location must be one of the following: Composite1024To512ImageInference.CENTER, Composite1024To512ImageInference.TOP_LEFT, Composite1024To512ImageInference.TOP_RIGHT, Composite1024To512ImageInference.BOTTOM_LEFT, Composite1024To512ImageInference.BOTTOM_RIGHT"
        assert logits.shape[0] == 512 and logits.shape[1] == 512 and logits.shape[2] == 3, \
            "logits must be of shape (512, 512, 3, k)"

        if location == self.CENTER:
            self.logits[self.CENTER_STR] = logits.detach().cpu().numpy()
            self.logits_obtained = True
        elif location == self.TOP_LEFT:
            self.logits[self.TOP_LEFT_STR] = logits[128:, 128:, ...].detach().cpu().numpy()
        elif location == self.TOP_RIGHT:
            self.logits[self.TOP_RIGHT_STR] = logits[128:, :-128, ...].detach().cpu().numpy()
        elif location == self.BOTTOM_LEFT:
            self.logits[self.BOTTOM_LEFT_STR] = logits[:-128, 128:, ...].detach().cpu().numpy()
        elif location == self.BOTTOM_RIGHT:
            self.logits[self.BOTTOM_RIGHT_STR] = logits[:-128, :-128, ...].detach().cpu().numpy()

    def store_logits_to_hdf(self, group: h5py.Group):
        if not self.logits_obtained:
            raise ValueError("Logits not obtained. Run inference first!")
        subgroup = group.create_group(self.tile_id)
        for key in self.logits:
            subgroup.create_dataset(key, data=self.logits[key], compression="gzip", compression_opts=9)

    def load_logits_from_hdf(self, group: h5py.Group, tile_id: str):
        self.tile_id = tile_id
        subgroup = group[self.tile_id]
        for key in subgroup:
            self.logits[key] = np.array(subgroup[key])
        self.logits_obtained = True

    def obtain_predictions(self, reduction_logit_average: bool=True, experts_only: bool=False):
        """
        Obtain the predictions (softmax probas) for the center 512x512 region of the 1024x1024 image.
        :param reduction_logit_average: if True, the reduction is done by first averaging the logits and computing the softmax
                Otherwise, the reduction is done by computing the softmax for each logit and then averaging
        :param experts_only: if True, only the experts are used for the prediction, if they are available.
                Experts predictions mean the center 256x256 region for the 768x768 image (as opposed to larger 512x512 region).
        :return a torch tensor of shape (512, 512, 3), containing softmax values for each pixel.
        """
        if not self.logits_obtained:
            raise ValueError("Logits not obtained. Run inference first!")
        if self.CENTER_STR not in self.logits:
            raise ValueError("ERROR: Center logits not obtained.")

        instances_array = np.zeros(shape=(4, 4), dtype=np.int32) # how many logit predictions in each 128x128 square in 512x512 region.
        stacked_instances_array = full_array(shape=(4, 4))

        # fill in the 512x512 square with logits
        for key in self.logits:
            if key == self.CENTER_STR:
                if experts_only:
                    for y in range(1, 3):
                        for x in range(1, 3):
                            instances_array[y, x] += 1
                            stacked_instances_array[y, x].append(self.logits[key][128 * y:128 * (y + 1),
                                                                 128 * x:128 * (x + 1), ...])
                else:
                    for y in range(0, 4):
                        for x in range(0, 4):
                            instances_array[y, x] += 1
                            stacked_instances_array[y, x].append(self.logits[key][128 * y:128 * (y + 1),
                                                                 128 * x:128 * (x + 1), ...])
            if key == self.TOP_LEFT_STR:
                if experts_only:
                    for y in range(0, 2):
                        for x in range(0, 2):
                            instances_array[y, x] += 1
                            stacked_instances_array[y, x].append(self.logits[key][128 * y:128 * (y + 1),
                                                                 128 * x:128 * (x + 1), ...])
                else:
                    for y in range(0, 3):
                        for x in range(0, 3):
                            instances_array[y, x] += 1
                            stacked_instances_array[y, x].append(self.logits[key][128 * y:128 * (y + 1),
                                                                 128 * x:128 * (x + 1), ...])
            if key == self.TOP_RIGHT_STR:
                if experts_only:
                    for y in range(0, 2):
                        for x in range(1, 3):
                            instances_array[y, x + 1] += 1
                            stacked_instances_array[y, x + 1].append(self.logits[key][128 * y:128 * (y + 1),
                                                                 128 * x:128 * (x + 1), ...])
                else:
                    for y in range(0, 3):
                        for x in range(0, 3):
                            instances_array[y, x + 1] += 1
                            stacked_instances_array[y, x + 1].append(self.logits[key][128 * y:128 * (y + 1),
                                                                     128 * x:128 * (x + 1), ...])
            if key == self.BOTTOM_LEFT_STR:
                if experts_only:
                    for y in range(1, 3):
                        for x in range(0, 2):
                            instances_array[y + 1, x] += 1
                            stacked_instances_array[y + 1, x].append(self.logits[key][128 * y:128 * (y + 1),
                                                                 128 * x:128 * (x + 1), ...])
                else:
                    for y in range(0, 3):
                        for x in range(0, 3):
                            instances_array[y + 1, x] += 1
                            stacked_instances_array[y + 1, x].append(self.logits[key][128 * y:128 * (y + 1),
                                                                     128 * x:128 * (x + 1), ...])
            if key == self.BOTTOM_RIGHT_STR:
                if experts_only:
                    for y in range(1, 3):
                        for x in range(1, 3):
                            instances_array[y + 1, x + 1] += 1
                            stacked_instances_array[y + 1, x + 1].append(self.logits[key][128 * y:128 * (y + 1),
                                                                 128 * x:128 * (x + 1), ...])
                else:
                    for y in range(0, 3):
                        for x in range(0, 3):
                            instances_array[y + 1, x + 1] += 1
                            stacked_instances_array[y + 1, x + 1].append(self.logits[key][128 * y:128 * (y + 1),
                                                                     128 * x:128 * (x + 1), ...])

        for y in range(0, 4):
            for x in range(0, 4):
                if instances_array[y, x] == 0:
                    stacked_instances_array[y, x].append(self.logits[self.CENTER_STR][128 * y:128 * (y + 1),
                                                                128 * x:128 * (x + 1), ...])

        # compute the prediction class now
        preds = []
        for y in range(0, 4):
            row_preds = []
            for x in range(0, 4):
                with torch.no_grad():
                    cat_tensor = torch.cat([torch.tensor(np_arr, device=config.device, dtype=torch.float32)
                               for np_arr in stacked_instances_array[y, x]], dim=-1)
                    if reduction_logit_average:
                        result = torch.softmax(torch.mean(cat_tensor, dim=-1), dim=-1)
                    else:
                        result = torch.mean(torch.softmax(cat_tensor, dim=-2), dim=-1)
                    row_preds.append(result)
            preds.append(torch.cat(row_preds, dim=1))
        return torch.cat(preds, dim=0)

    def __del__(self):
        if self.image_loaded:
            del self.image, self.region_mask
            self.logits.clear()

if __name__ == "__main__":
    image, region_mask = load_image("0033bbc76b6b")
    cv2.imshow("image", image)
    cv2.waitKey(0)

    cv2.imshow("region_mask", region_mask.astype(np.uint8) * 255)
    cv2.waitKey(0)