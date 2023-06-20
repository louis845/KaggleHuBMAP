import gc
import os

import numpy as np
import pandas as pd
import h5py
import cv2
import tqdm
import torch

import model_data_manager
import obtain_reconstructed_wsi_images
import obtain_reconstructed_binary_segmentation
import config

folder = "reconstructed_wsi_data"

def obtain_relative_bounding_box(mask: np.ndarray):
    assert mask.ndim == 2, "Mask must be 2D"
    assert mask.dtype == bool, "Mask must be boolean"

    exists_x = np.argwhere(np.any(mask, axis=0)).squeeze(-1)
    exists_y = np.argwhere(np.any(mask, axis=1)).squeeze(-1)

    bounds_x1, bounds_x2 = np.min(exists_x), np.max(exists_x) + 1
    bounds_y1, bounds_y2 = np.min(exists_y), np.max(exists_y) + 1

    return bounds_x1, bounds_y1, bounds_x2, bounds_y2

def obtain_top_clearance(mask: np.ndarray, x: int, y: int):
    assert 0 <= x < mask.shape[1], "x must be within the mask"
    assert 0 <= y < mask.shape[0], "y must be within the mask"
    assert mask[y, x], "The pixel must be within the mask"

    top_values = np.argwhere(np.logical_not(mask[:y, x])).squeeze(-1)
    if len(top_values) == 0:
        return y
    else:
        return y - top_values.max() - 1

def obtain_bottom_clearance(mask: np.ndarray, x: int, y: int):
    assert 0 <= x < mask.shape[1], "x must be within the mask"
    assert 0 <= y < mask.shape[0], "y must be within the mask"
    assert mask[y, x], "The pixel must be within the mask"

    bottom_values = np.argwhere(np.logical_not(mask[y:, x])).squeeze(-1)

    if len(bottom_values) == 0:
        return mask.shape[0] - y
    else:
        return bottom_values.min()

rng = np.random.default_rng()

class Region:
    def __init__(self, wsi_id: int, hdf5_group: h5py.Group, writable=False):
        self.wsi_id = wsi_id

        self.hdf5_group = hdf5_group
        self.writable = writable

        # the following arrays are stored as HDF5 datasets
        self.region = None # boolean array of shape (height, width) defining the region mask

        # number of pixels available to the top and bottom of the region
        # a int array of shape (height, width, 2), where [:, :, 0] indicates how many pixels available in the top, and
        # [:, :, 1] indicates how many pixels available in the bottom. This means [y, x, y - [y, x, 0] : y + [y, x, 1]]
        # are all True in the region mask.
        self.horizontal_clearance = None

        # see generate_interior_pixels
        self.interior_pixels = None
        self.interior_pixels_list = None # a list of interior pixels, shape (N, 2). Each row is (y, x)
        self.interior_box_width = None # An integer. see generate_interior_pixels

        self.rotated_interior_pixels = None
        self.rotated_interior_pixels_list = None
        self.rotated_interior_box_width = None

    def generate_region(self, wsi_tile_subset=None):
        """Compute the region mask for this wsi. If wsi_tile_subset is not None, then only the tiles in the subset will be considered."""
        assert self.writable, "This Region object is not writable"

        # Generate the region mask first
        wsi_information = model_data_manager.data_information
        wsi_information = wsi_information.loc[wsi_information["source_wsi"] == self.wsi_id]
        width = int(wsi_information["i"].max() + 512)
        height = int(wsi_information["j"].max() + 512)

        region = np.zeros((height, width), dtype=bool)

        for wsi_tile in wsi_information.index:
            if wsi_tile_subset is None or wsi_tile in wsi_tile_subset:
                x = wsi_information.loc[wsi_tile, "i"]
                y = wsi_information.loc[wsi_tile, "j"]

                region[y:y + 512, x:x + 512] = True

        # Compute the horizontal clearance
        horizontal_clearance = np.full((height, width, 2), dtype=np.int32, fill_value=-1)

        available_row_indices = np.unique(np.squeeze(np.argwhere(np.any(region, axis=1)), -1))
        for y in tqdm.tqdm(available_row_indices):
            x_indices = np.unique(np.squeeze(np.argwhere(region[y, :]), -1))
            for x in x_indices:
                if y > 0 and region[y - 1, x]:
                    horizontal_clearance[y, x, 0] = horizontal_clearance[y - 1, x, 0] + 1
                    horizontal_clearance[y, x, 1] = horizontal_clearance[y - 1, x, 1] - 1
                else:
                    horizontal_clearance[y, x, 0] = obtain_top_clearance(region, x, y)
                    horizontal_clearance[y, x, 1] = obtain_bottom_clearance(region, x, y)

        # Save to hdf5
        self.hdf5_group.create_dataset("wsi_id", data=self.wsi_id, dtype=np.int32)
        self.horizontal_clearance = self.hdf5_group.create_dataset("horizontal_clearance", data=horizontal_clearance, dtype=np.int32, compression="gzip", compression_opts=9, shape=(height, width, 2))
        self.region = self.hdf5_group.create_dataset("region", data=region, dtype=bool, compression="gzip", compression_opts=9, shape=(height, width))

        del region
        del horizontal_clearance
        gc.collect()

    def generate_interior_pixels(self, interior_box: int=512):
        """Compute the pixels so that the interior box with specified size is completely within the region mask.
        The interior box should be even, and is centered within each interior pixels. More precisely, given a pixel (x,y),
        the interior box is [y - interior_box//2 : y + interior_box//2, x - interior_box//2 : x + interior_box//2].
        In simple terms, the pixel (x,y) is the top left corner of the bottom right quadrant of the interior box."""

        assert self.writable, "This Region object is not writable"
        assert self.region is not None, "The region mask must be generated first, either by generate_region or load_from_hdf5"
        assert self.horizontal_clearance is not None, "The horizontal clearance must be generated first, either by generate_region or load_from_hdf5"
        assert interior_box % 2 == 0, "The interior box size must be even"

        interior_radius = interior_box // 2

        region = np.array(self.region, dtype=bool)
        horizontal_clearance = np.array(self.horizontal_clearance, dtype=np.int32)

        available_row_indices = np.unique(np.squeeze(np.argwhere(np.any(region, axis=1)), -1))

        print("Computing interior pixels...")
        interior_pixels = np.zeros(self.region.shape, dtype=bool)
        for y in tqdm.tqdm(available_row_indices):
            x_indices = np.unique(np.squeeze(np.argwhere(region[y, :]), -1))
            for x in x_indices:
                if y >= interior_radius and y < horizontal_clearance.shape[0] - interior_radius and x >= interior_radius and x < horizontal_clearance.shape[1] - interior_radius:
                    if interior_pixels[y - 1, x]:
                        interior_pixels[y, x] = np.all(region[y + interior_radius - 1, x - interior_radius:x + interior_radius])
                    elif interior_pixels[y, x - 1]:
                        interior_pixels[y, x] = np.all(region[y - interior_radius:y + interior_radius, x + interior_radius - 1])
                    else:
                        interior_pixels[y, x] = np.all(horizontal_clearance[y, x - interior_radius:x + interior_radius, :] >= interior_radius)

        self.interior_pixels = self.hdf5_group.create_dataset("interior_pixels", data=interior_pixels, dtype=bool, compression="gzip", compression_opts=9, shape=self.region.shape)
        interior_pixels_list = np.argwhere(interior_pixels)
        self.interior_pixels_list = self.hdf5_group.create_dataset("interior_pixels_list", data=interior_pixels_list, dtype=np.int32, compression="gzip", compression_opts=9, shape=(len(interior_pixels_list), 2))
        self.interior_box_width = interior_box
        self.hdf5_group.create_dataset("interior_box_width", data=interior_box, dtype=np.int32)
        del interior_pixels, interior_pixels_list
        gc.collect()


        print("Computing rotated interior pixels...")
        rotated_interior_pixels = np.zeros(self.region.shape, dtype=bool)
        interior_radius_rotated = int(np.ceil(interior_radius * np.sqrt(2)))
        for y in tqdm.tqdm(available_row_indices):
            x_indices = np.unique(np.squeeze(np.argwhere(region[y, :]), -1))
            for x in x_indices:
                if y >= interior_radius_rotated and y < horizontal_clearance.shape[0] - interior_radius_rotated and x >= interior_radius_rotated and x < horizontal_clearance.shape[1] - interior_radius_rotated:
                    if rotated_interior_pixels[y - 1, x]:
                        rotated_interior_pixels[y, x] = np.all(region[y + interior_radius_rotated - 1, x - interior_radius_rotated:x + interior_radius_rotated])
                    elif rotated_interior_pixels[y, x - 1]:
                        rotated_interior_pixels[y, x] = np.all(region[y - interior_radius_rotated:y + interior_radius_rotated, x + interior_radius_rotated - 1])
                    else:
                        rotated_interior_pixels[y, x] = np.all(horizontal_clearance[y, x - interior_radius_rotated:x + interior_radius_rotated, :] >= interior_radius_rotated)

        self.rotated_interior_pixels = self.hdf5_group.create_dataset("rotated_interior_pixels", data=rotated_interior_pixels, dtype=bool,
                                                              compression="gzip", compression_opts=9, shape=self.region.shape)
        rotated_interior_pixels_list = np.argwhere(rotated_interior_pixels)
        self.rotated_interior_pixels_list = self.hdf5_group.create_dataset("rotated_interior_pixels_list", data=rotated_interior_pixels_list, dtype=np.int32,
                                                                   compression="gzip", compression_opts=9, shape=(len(rotated_interior_pixels_list), 2))
        self.rotated_interior_box_width = interior_radius_rotated * 2
        self.hdf5_group.create_dataset("rotated_interior_box_width", data=self.rotated_interior_box_width, dtype=np.int32)
        del rotated_interior_pixels, rotated_interior_pixels_list
        gc.collect()

    def save_interior_pixels_image(self, image_name, overlay=None):
        assert self.interior_pixels is not None, "The interior pixels must be generated first, either by generate_interior_pixels or load_from_hdf5"

        interior_pixels = np.repeat(np.expand_dims(np.array(self.interior_pixels, dtype=np.uint8) * 255, axis=-1), axis=-1, repeats=3)

        if overlay is not None:
            interior_pixels = cv2.addWeighted(interior_pixels, 0.5, overlay, 0.5, 0)

        obtain_reconstructed_wsi_images.draw_grids(interior_pixels)


        # Save image as PNG grayscale
        cv2.imwrite(os.path.join(folder, "{}.png".format(image_name)), interior_pixels)

    def save_rotated_interior_pixels_image(self, image_name, overlay=None):
        assert self.rotated_interior_pixels is not None, "The interior pixels must be generated first, either by generate_interior_pixels or load_from_hdf5"

        r_interior_pixels = np.repeat(np.expand_dims(np.array(self.rotated_interior_pixels, dtype=np.uint8) * 255, axis=-1), axis=-1, repeats=3)

        if overlay is not None:
            r_interior_pixels = cv2.addWeighted(r_interior_pixels, 0.5, overlay, 0.5, 0)

        obtain_reconstructed_wsi_images.draw_grids(r_interior_pixels)


        # Save image as PNG grayscale
        cv2.imwrite(os.path.join(folder, "{}.png".format(image_name)), r_interior_pixels)

    def save_region_image(self, image_name):
        assert self.region is not None, "The region mask must be generated first, either by generate_region or load_from_hdf5"

        region = np.repeat(np.expand_dims(np.array(self.region, dtype=np.uint8) * 255, axis=-1), axis=-1, repeats=3)

        obtain_reconstructed_wsi_images.draw_grids(region)

        # Save image as PNG grayscale
        cv2.imwrite(os.path.join(folder, "{}.png".format(image_name)), region)

    def save_horizontal_clearance_image(self, image_name):
        assert self.horizontal_clearance is not None, "The horizontal clearance must be generated first, either by generate_region or load_from_hdf5"

        horizontal_clearance = np.array(self.horizontal_clearance, dtype=np.float32)
        horizontal_clearance_top = np.repeat(np.expand_dims(horizontal_clearance[:, :, 0] * 126.0 / np.max(horizontal_clearance[:, :, 0]), axis=-1), axis=-1, repeats=3).astype(np.uint8)
        horizontal_clearance_bottom = np.repeat(np.expand_dims(horizontal_clearance[:, :, 1] * 126.0 / np.max(horizontal_clearance[:, :, 1]), axis=-1), axis=-1, repeats=3).astype(np.uint8)

        obtain_reconstructed_wsi_images.draw_grids(horizontal_clearance_top)
        obtain_reconstructed_wsi_images.draw_grids(horizontal_clearance_bottom)

        # Save image as PNG grayscale
        cv2.imwrite(os.path.join(folder, "{}_top.png".format(image_name)), horizontal_clearance_top)
        cv2.imwrite(os.path.join(folder, "{}_bottom.png".format(image_name)), horizontal_clearance_bottom)

    def load_from_hdf5(self):
        assert self.wsi_id == int(self.hdf5_group["wsi_id"][()])
        self.horizontal_clearance = self.hdf5_group["horizontal_clearance"]
        self.region = self.hdf5_group["region"]

        if "interior_pixels" in self.hdf5_group:
            self.interior_pixels = self.hdf5_group["interior_pixels"]
            self.interior_pixels_list = self.hdf5_group["interior_pixels_list"]
            self.interior_box_width = self.hdf5_group["interior_box_width"][()]

    def sample_interior_pixels(self, num_samples):
        indices = np.unique(np.random.choice(self.interior_pixels_list.shape[0], num_samples, replace=False))

        return np.array(self.interior_pixels_list[indices, :], dtype=np.int32)[..., ::-1]

    def get_region_mask(self, x1, x2, y1, y2):
        return np.array(self.region[y1:y2, x1:x2])

class WSIImage:
    """Wrapper class for storing all WSI images."""
    def __init__(self):
        if not os.path.isfile(os.path.join(folder, "wsi_images.hdf5")):
            print()
            print("WSI images not found. Generating...")
            with h5py.File(os.path.join(folder, "wsi_images.hdf5"), "w") as f:
                for wsi_id in tqdm.tqdm(range(1, 15)):
                    if wsi_id != 5:
                        wsi_image_np = obtain_reconstructed_wsi_images.construct_wsi(wsi_id)
                        f.create_dataset("wsi_{}".format(wsi_id), data=wsi_image_np, dtype=np.uint8, compression="gzip", compression_opts=9)
                        del wsi_image_np

            print("Successfully generated WSI images.")

        self.wsi_images = h5py.File(os.path.join(folder, "wsi_images.hdf5"), "r")

    def get_whole_image(self, wsi_id):
        return np.array(self.wsi_images["wsi_{}".format(wsi_id)], dtype=np.uint8)

    def get_image(self, wsi_id, x1, x2, y1, y2):
        return np.array(self.wsi_images["wsi_{}".format(wsi_id)][y1:y2, x1:x2, :], dtype=np.uint8)

    def __del__(self):
        self.wsi_images.close()


images = WSIImage()
class ImageSampler:
    def __init__(self, wsi_region: Region, sampling_region: Region, polygon_masks: obtain_reconstructed_binary_segmentation.WSIMask, wsi_id: int):
        assert sampling_region.interior_pixels_list is not None, "The interior pixels of the sampling region must be generated first! Use generate_interior_pixels()"
        assert wsi_region.region is not None, "The region mask of the WSI region must be generated first! Use generate_region()"

        self.wsi_region = wsi_region
        self.sampling_region = sampling_region
        self.polygons = polygon_masks

    def sample_interior_pixels(self, num_samples):
        return self.sampling_region.sample_interior_pixels(num_samples)

    def obtain_image(self, x, y, image_width: int):
        """Obtains the image at the specified location. The x, y coordinates are the top left corner of the bottom right
            quadrant of the subimage."""
        assert image_width % 2 == 0, "The image width must be an even number!"

        x1, x2 = x - image_width // 2, x + image_width // 2
        y1, y2 = y - image_width // 2, y + image_width // 2

        x1 = max(0, x1)
        x2 = min(self.wsi_region.region.shape[1], x2)
        y1 = max(0, y1)
        y2 = min(self.wsi_region.region.shape[0], y2)

        image = torch.tensor(images.get_image(self.wsi_region.wsi_id, x1, x2, y1, y2), dtype=torch.float32, device=config.device)
        ground_truth = torch.tensor(self.polygons.obtain_blood_vessel_mask(x1, x2, y1, y2), dtype=torch.long, device=config.device)
        region_mask = torch.tensor(self.wsi_region.get_region_mask(x1, x2, y1, y2), dtype=torch.bool, device=config.device)

        return image, ground_truth, region_mask

    def obtain_image_with_augmentation(self, x, y, image_width: int):
        rotation = rng.uniform(0, 360)

        image, ground_truth, region_mask = self.obtain_image(x, y, image_width)


if not os.path.isdir(folder):
    os.mkdir(folder)

if not os.path.isfile(os.path.join(folder, "data.hdf5")):
    print("Data not yet generated. Generating data now....")
    with h5py.File(os.path.join(folder, "data.hdf5"), "w") as segmentation_masks:
        for wsi_id in range(1, 3):
            print("Generating data for wsi_{}".format(wsi_id))
            group = segmentation_masks.create_group("wsi_{}".format(wsi_id))
            r = Region(wsi_id, group, writable=True)
            r.generate_region()
            r.generate_interior_pixels()
            del r

with h5py.File(os.path.join(folder, "data.hdf5"), "r") as segmentation_masks:
    for wsi_id in range(1, 3):
        print("Generating images for wsi_{}".format(wsi_id))
        group = segmentation_masks["wsi_{}".format(wsi_id)]
        r = Region(wsi_id, group, writable=False)
        r.load_from_hdf5()
        background = images.get_whole_image(wsi_id)[..., ::-1]
        r.save_interior_pixels_image("wsi512_{}".format(wsi_id), background)
        r.save_rotated_interior_pixels_image("wsi512_{}_rotated".format(wsi_id), background)