import gc
import os
import time

import numpy as np
import pandas as pd
import h5py
import cv2
import tqdm
import torch
import torchvision.transforms.functional

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

def argmax_value2d(tensor: torch.Tensor):
    max_index = torch.argmax(tensor)

    i = max_index // tensor.shape[1]
    j = max_index % tensor.shape[1]

    return i, j

def obtain_mask_clearance_optimal_shape(radius: int, min_radius: int, clearance_values: torch.Tensor, device=config.device):
    top_clearance_values = torch.cummin(torch.flip(clearance_values[:radius], dims=[0]), dim=0)[0]
    bottom_clearance_values = torch.cummin(clearance_values[radius:], dim=0)[0]

    joint_clearance = torch.min(top_clearance_values.unsqueeze(-1), bottom_clearance_values)
    joint_height = torch.arange(1, radius + 1, device=device, dtype=torch.int32).unsqueeze(-1) + \
                   torch.arange(1, radius + 1, device=device, dtype=torch.int32)

    joint_area_clearance = joint_height * joint_clearance
    joint_area_clearance[:, :(min_radius - 1)] = -1
    joint_area_clearance[:(min_radius - 1), :] = -1
    joint_area_height = torch.clone(joint_area_clearance)

    joint_area_clearance[joint_height > joint_clearance] = -1
    joint_area_height[joint_height < joint_clearance] = -1

    top1, bottom1 = argmax_value2d(joint_area_clearance)
    top2, bottom2 = argmax_value2d(joint_area_height)

    width1 = joint_clearance[top1, bottom1]
    width2 = joint_clearance[top2, bottom2]

    if top1.item() == bottom1.item() == 0:
        top1 = min_radius - 1
        bottom1 = min_radius - 1
        width1 = min_radius

    return top1, bottom1, width1, top2, bottom2, width2

def obtain_mask_clearance(mask: torch.Tensor, min_radius: int, device=config.device):
    assert mask.dtype == torch.bool, "Mask must be boolean"
    assert mask.shape[0] == mask.shape[1], "Mask must be square"
    assert mask.shape[0] % 2 == 0, "Mask must have even size"

    """Obtain a 'maximum' cross bounding box for the mask tensor. This means that the bounding box is the smallest
       'cross shape' that contains all the True values in the mask tensor, which contains the center of the image."""
    radius = mask.shape[0] // 2

    # first compute left clearance values and right clearance values.
    # they are of shape (height,), storing how many pixels on the left/right are such that the mask is True.
    left_clearance = torch.argwhere(torch.logical_not(mask[:, :radius]))
    right_clearance = torch.argwhere(torch.logical_not(mask[:, radius:]))

    left_clearance_row = left_clearance[:, 0]
    right_clearance_row = right_clearance[:, 0]

    dummy_true = torch.tensor([True], dtype=torch.bool, device=device)

    left_clearance_values = torch.full(size=(mask.shape[0],), dtype=torch.long, device=device, fill_value=radius)
    if left_clearance_row.shape[0] > 0:
        left_clearance = left_clearance[torch.concat([left_clearance_row[1:] > left_clearance_row[:-1], dummy_true]), :]
        left_clearance_values[left_clearance[:, 0]] = radius - left_clearance[:, 1] - 1

    right_clearance_values = torch.full(size=(mask.shape[0],), dtype=torch.long, device=device, fill_value=radius)
    if right_clearance_row.shape[0] > 0:
        right_clearance = right_clearance[torch.concat([dummy_true, right_clearance_row[1:] > right_clearance_row[:-1]]), :]
        right_clearance_values[right_clearance[:, 0]] = right_clearance[:, 1]

    # now compute the cross mask
    cross_mask = torch.zeros(size=(mask.shape[0], mask.shape[1]), dtype=torch.bool, device=device)
    top1, bottom1, width1, top2, bottom2, width2 = obtain_mask_clearance_optimal_shape(radius, min_radius, left_clearance_values, device=device)
    cross_mask[radius - 1 - top1:radius + 1 + bottom1, radius - width1:radius] = True
    cross_mask[radius - 1 - top2:radius + 1 + bottom2, radius - width2:radius] = True
    top1, bottom1, width1, top2, bottom2, width2 = obtain_mask_clearance_optimal_shape(radius, min_radius, right_clearance_values, device=device)
    cross_mask[radius - 1 - top1:radius + 1 + bottom1, radius:radius + width1] = True
    cross_mask[radius - 1 - top2:radius + 1 + bottom2, radius:radius + width2] = True

    return cross_mask

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
        self.horizontal_clearance = self.hdf5_group.create_dataset("horizontal_clearance", data=horizontal_clearance, dtype=np.int32, compression="gzip", compression_opts=9, shape=(height, width, 2), chunks=(512, 512, 2))
        self.region = self.hdf5_group.create_dataset("region", data=region, dtype=bool, compression="gzip", compression_opts=9, shape=(height, width), chunks=(512, 512))

        del region
        del horizontal_clearance
        gc.collect()

    def generate_interior_pixels(self, interior_box: int=512, wsi_tile_write=None):
        """Compute the pixels so that the interior box with specified size is completely within the region mask.
        The interior box should be even, and is centered within each interior pixels. More precisely, given a pixel (x,y),
        the interior box is [y - interior_box//2 : y + interior_box//2, x - interior_box//2 : x + interior_box//2].
        In simple terms, the pixel (x,y) is the top left corner of the bottom right quadrant of the interior box.

        WSI tile write is the rainbow table of argwhere locations of the pixels in the interior and also in the tile."""

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

        self.interior_pixels = self.hdf5_group.create_dataset("interior_pixels", data=interior_pixels, dtype=bool, compression="gzip", compression_opts=9, shape=self.region.shape, chunks=(512, 512))
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
                                                              compression="gzip", compression_opts=9, shape=self.region.shape, chunks=(512, 512))
        rotated_interior_pixels_list = np.argwhere(rotated_interior_pixels)
        self.rotated_interior_pixels_list = self.hdf5_group.create_dataset("rotated_interior_pixels_list", data=rotated_interior_pixels_list, dtype=np.int32,
                                                                   compression="gzip", compression_opts=9, shape=(len(rotated_interior_pixels_list), 2))
        self.rotated_interior_box_width = interior_radius_rotated * 2
        self.hdf5_group.create_dataset("rotated_interior_box_width", data=self.rotated_interior_box_width, dtype=np.int32)
        del rotated_interior_pixels, rotated_interior_pixels_list
        gc.collect()

        if wsi_tile_write is not None:
            print("Computing interior pixels rainbow table...")
            for tile_id in wsi_tile_write:
                assert model_data_manager.data_information.loc[
                           tile_id, "source_wsi"] == self.wsi_id, "The tile_id must belong to the current wsi_id!"

                x = model_data_manager.data_information.loc[tile_id, "i"]
                y = model_data_manager.data_information.loc[tile_id, "j"]

                pixels = self.get_interior_pixels_mask(x, x + 512, y, y + 512)
                locs = (np.argwhere(pixels) + np.array([[y, x]], dtype=np.int64)).astype(np.int32)
                self.hdf5_group.create_dataset("{}_table".format(tile_id), data=locs, dtype=np.int32, compression="gzip", compression_opts=9, shape=locs.shape)

                del pixels, locs
                gc.collect()


    def save_interior_pixels_image(self, image_path, overlay=None):
        assert self.interior_pixels is not None, "The interior pixels must be generated first, either by generate_interior_pixels or load_from_hdf5"

        interior_pixels = np.repeat(np.expand_dims(np.array(self.interior_pixels, dtype=np.uint8) * 255, axis=-1), axis=-1, repeats=3)

        if overlay is not None:
            interior_pixels = cv2.addWeighted(interior_pixels, 0.5, overlay, 0.5, 0)

        obtain_reconstructed_wsi_images.draw_grids(interior_pixels)

        # Save image as PNG grayscale
        cv2.imwrite(image_path, interior_pixels)

    def save_rotated_interior_pixels_image(self, image_path, overlay=None):
        assert self.rotated_interior_pixels is not None, "The interior pixels must be generated first, either by generate_interior_pixels or load_from_hdf5"

        r_interior_pixels = np.repeat(np.expand_dims(np.array(self.rotated_interior_pixels, dtype=np.uint8) * 255, axis=-1), axis=-1, repeats=3)

        if overlay is not None:
            r_interior_pixels = cv2.addWeighted(r_interior_pixels, 0.5, overlay, 0.5, 0)

        obtain_reconstructed_wsi_images.draw_grids(r_interior_pixels)

        # Save image as PNG grayscale
        cv2.imwrite(image_path, r_interior_pixels)

    def save_region_image(self, image_path, overlay=None):
        assert self.region is not None, "The region mask must be generated first, either by generate_region or load_from_hdf5"

        region = np.repeat(np.expand_dims(np.array(self.region, dtype=np.uint8) * 255, axis=-1), axis=-1, repeats=3)

        if overlay is not None:
            region = cv2.addWeighted(region, 0.5, overlay, 0.5, 0)

        obtain_reconstructed_wsi_images.draw_grids(region)

        # Save image as PNG grayscale
        cv2.imwrite(image_path, region)

    def save_horizontal_clearance_image(self, image_path):
        assert self.horizontal_clearance is not None, "The horizontal clearance must be generated first, either by generate_region or load_from_hdf5"

        horizontal_clearance = np.array(self.horizontal_clearance, dtype=np.float32)
        horizontal_clearance_top = np.repeat(np.expand_dims(horizontal_clearance[:, :, 0] * 126.0 / np.max(horizontal_clearance[:, :, 0]), axis=-1), axis=-1, repeats=3).astype(np.uint8)
        horizontal_clearance_bottom = np.repeat(np.expand_dims(horizontal_clearance[:, :, 1] * 126.0 / np.max(horizontal_clearance[:, :, 1]), axis=-1), axis=-1, repeats=3).astype(np.uint8)

        obtain_reconstructed_wsi_images.draw_grids(horizontal_clearance_top)
        obtain_reconstructed_wsi_images.draw_grids(horizontal_clearance_bottom)

        # Save image as PNG grayscale
        cv2.imwrite(image_path, horizontal_clearance_top)
        cv2.imwrite(image_path, horizontal_clearance_bottom)

    def load_from_hdf5(self):
        assert self.wsi_id == int(self.hdf5_group["wsi_id"][()])
        self.horizontal_clearance = self.hdf5_group["horizontal_clearance"]
        self.region = self.hdf5_group["region"]

        if "interior_pixels" in self.hdf5_group:
            self.interior_pixels = self.hdf5_group["interior_pixels"]
            self.interior_pixels_list = self.hdf5_group["interior_pixels_list"]
            self.interior_box_width = self.hdf5_group["interior_box_width"][()]

            self.rotated_interior_pixels = self.hdf5_group["rotated_interior_pixels"]
            self.rotated_interior_pixels_list = self.hdf5_group["rotated_interior_pixels_list"]
            self.rotated_interior_box_width = self.hdf5_group["rotated_interior_box_width"][()]

    def sample_interior_pixels(self, num_samples):
        indices = np.unique(np.random.choice(self.interior_pixels_list.shape[0], num_samples, replace=False))

        return np.array(self.interior_pixels_list[indices, :], dtype=np.int32)[..., ::-1]

    def sample_interior_pixels_wsi(self, tile_id):
        locations = self.hdf5_group["{}_table".format(tile_id)]
        pixel = locations[np.random.randint(0, locations.shape[0]), :]
        return pixel[1], pixel[0]

    def get_region_mask(self, x1, x2, y1, y2):
        return np.array(self.region[y1:y2, x1:x2])

    def get_interior_pixels_mask(self, x1, x2, y1, y2):
        return np.array(self.interior_pixels[y1:y2, x1:x2])

    def pixel_in_interior(self, x, y) -> bool:
        return self.interior_pixels[y, x]

    def pixel_in_rotated_interior(self, x, y) -> bool:
        return self.rotated_interior_pixels[y, x]

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
                        f.create_dataset("wsi_{}".format(wsi_id), data=wsi_image_np, dtype=np.uint8, compression="gzip", compression_opts=9, chunks=(512, 512, 3))
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
    def __init__(self, wsi_region: Region, sampling_region: Region, polygon_masks: obtain_reconstructed_binary_segmentation.WSIMask, image_width: int, device=config.device):
        """
        :param wsi_region: The region representing the available image pixels in the WSI.
        :param sampling_region: The region representing the available ground truth mask pixels in the WSI. To enable train/test split.
        :param polygon_masks: The WSIMask object (from obtain_reconstructed_binary_segmentation) containing the polygon masks.
        :param image_width: The width of the image to be sampled.
        """
        assert sampling_region.interior_pixels_list is not None, "The interior pixels of the sampling region must be generated first! Use generate_interior_pixels()"
        assert wsi_region.region is not None, "The region mask of the WSI region must be generated first! Use generate_region()"
        assert wsi_region.wsi_id == sampling_region.wsi_id, "The WSI IDs of the WSI region and sampling region must be the same!"
        assert image_width % 2 == 0, "The image width must be even!"

        self.wsi_region = wsi_region
        self.sampling_region = sampling_region
        self.polygons = polygon_masks
        self.wsi_id = wsi_region.wsi_id
        self.image_width = image_width

        image_radius = image_width // 2
        self.prediction_radius = self.sampling_region.interior_box_width // 2
        self.center_mask = torch.zeros((image_width, image_width), dtype=torch.bool, device=device)
        self.center_mask[image_radius - self.prediction_radius:image_radius + self.prediction_radius, image_radius - self.prediction_radius:image_radius + self.prediction_radius] = True
        self.device = device

    def sample_interior_pixels(self, num_samples):
        return self.sampling_region.sample_interior_pixels(num_samples)

    def obtain_image(self, x, y, image_width: int, device=config.device):
        """Obtains the image at the specified location. The x, y coordinates are the top left corner of the bottom right
            quadrant of the subimage.

            Returns:
                image: The rgb image of shape (3, image_width, image_width)
                region_mask: The region mask of shape (image_width, image_width), representing the pixels in the WSI that are available.
                ground_truth: The ground truth mask of shape (image_width, image_width), representing the semantic classes.
                ground_truth_mask: The ground truth mask of shape (image_width, image_width), representing the pixels that are available for the semantic classes.
            The images are converted to torch.float32 type and stacked together to form a (6, image_width, image_width) tensor."""
        assert image_width % 2 == 0, "The image width must be an even number!"

        x1, x2 = x - image_width // 2, x + image_width // 2
        y1, y2 = y - image_width // 2, y + image_width // 2

        x1_int = max(0, x1)
        x2_int = min(self.wsi_region.region.shape[1], x2)
        y1_int = max(0, y1)
        y2_int = min(self.wsi_region.region.shape[0], y2)

        with torch.no_grad():
            image = images.get_image(self.wsi_region.wsi_id, x1_int, x2_int, y1_int, y2_int).astype(dtype=np.float32).transpose([2, 0, 1]) # float32, (0-2)
            region_mask = self.wsi_region.get_region_mask(x1_int, x2_int, y1_int, y2_int).astype(dtype=np.float32) # bool (3)
            ground_truth = self.polygons.obtain_blood_vessel_mask(x1_int, x2_int, y1_int, y2_int).astype(dtype=np.float32) # long (4)
            gt1 = self.sampling_region.get_region_mask(x1_int, x2_int, y1_int, y2_int)
            gt2 = self.wsi_region.get_interior_pixels_mask(x1_int, x2_int, y1_int, y2_int)

            ground_truth_mask = np.logical_and(gt1, gt2).astype(dtype=np.float32) # bool (5)


            cat = np.concatenate([image, np.expand_dims(region_mask, axis=0), np.expand_dims(ground_truth, axis=0), np.expand_dims(ground_truth_mask, axis=0)], axis=0)
            cat = torch.tensor(cat, dtype=torch.float32, device=device)
            cat[4, ...] = cat[4, ...] * cat[5, ...]

            cat = torch.nn.functional.pad(cat, (x1_int - x1, x2 - x2_int, y1_int - y1, y2 - y2_int))

        return cat

    def obtain_image_with_augmentation(self, x, y, augmentation: bool=True):
        """Returns:
        torch.cat([image, region_mask], dim=0): The image and region mask of shape (4, image_width, image_width), where image is rgb image and region_mask is the 0-1 region mask.
        ground_truth: A long tensor representing the ground truth semantic labels.
        ground_truth_mask: A bool tensor representing the ground truth mask.
        """
        device = self.device

        image_width = self.image_width
        assert image_width % 2 == 0, "The image width must be an even number!"
        with torch.no_grad():
            image_radius = image_width // 2
            if self.sampling_region.pixel_in_rotated_interior(x, y) and augmentation:
                rotation = rng.uniform(0, 360)

                sample_image_radius = int(image_radius * (np.sin(np.deg2rad(rotation % 90)) + np.cos(np.deg2rad(rotation % 90))))

                cat = self.obtain_image(x, y, sample_image_radius * 2, device=device) # image (0-2), region_mask (3), ground_truth (4), ground_truth_mask (5)


                cat = torchvision.transforms.functional.rotate(cat, angle=rotation, fill=0.0) \
                    [:, sample_image_radius - image_radius:sample_image_radius + image_radius, sample_image_radius - image_radius:sample_image_radius + image_radius]

                region_mask = obtain_mask_clearance(cat[3, ...].to(torch.bool), min_radius=self.prediction_radius, device=device)

                cat *= region_mask
            else:
                cat = self.obtain_image(x, y, image_width, device=device)

                if augmentation:
                    rotation = np.random.randint(0, 4) * 90
                    cat = torchvision.transforms.functional.rotate(cat, angle=rotation)

            # flip the last dimension with a 50% chance
            if augmentation and (rng.uniform(0, 1) > 0.5):
                cat = torch.flip(cat, dims=[-1])

            # restrict ground_truth and ground_truth_mask to the center pixels.
            cat[4:6, ...] = cat[4:6, ...] * self.center_mask

            # randomly dropout corner pixels.
            if augmentation:
                dropouts = (rng.beta(0.7, 1.0, size=(8,)) * (image_radius - self.sampling_region.interior_box_width // 2)).astype(dtype=np.int32)
                cat[:4, :dropouts[0], :dropouts[1]] = 0.0
                cat[:4, :dropouts[2], -dropouts[3]:] = 0.0
                cat[:4, -dropouts[4]:, :dropouts[5]] = 0.0
                cat[:4, -dropouts[6]:, -dropouts[7]:] = 0.0

        ret = (cat[:4, ...], cat[4, ...].to(torch.long), cat[5, ...])
        return ret

    def obtain_random_sample_pixel_from_tile(self, tile_id: str):
        assert model_data_manager.data_information.loc[tile_id, "source_wsi"] == self.wsi_id, "The tile_id must belong to the current wsi_id!"

        return self.sampling_region.sample_interior_pixels_wsi(tile_id)


    def obtain_random_image_from_tile(self, tile_id: str, augmentation: bool=True):
        x, y = self.obtain_random_sample_pixel_from_tile(tile_id)

        with torch.no_grad():
            return self.obtain_image_with_augmentation(x, y, augmentation=augmentation)

    def obtain_deep_supervision_outputs(self, ground_truth, ground_truth_mask, deep_supervision_downsamples=0):
        """Assumes that ground_truth is long tensor of shape (image_width, image_width), and ground_truth_mask is a float32 tensor of shape (image_width, image_width)"""
        ground_truth_deep = []
        ground_truth_mask_deep = []
        with torch.no_grad():
            for k in range(deep_supervision_downsamples):
                scale_factor = 2 ** (k + 1)
                ground_truth_deep.append(
                    torch.nn.functional.max_pool2d(ground_truth.view(1, 1, self.image_width, self.image_width).to(torch.float32), kernel_size=scale_factor, stride=scale_factor).to(torch.long)\
                        .view(self.image_width // scale_factor, self.image_width // scale_factor)
                )
                ground_truth_mask_deep.append(torch.nn.functional.max_pool2d(ground_truth_mask.view(1, 1, self.image_width, self.image_width), kernel_size=scale_factor, stride=scale_factor))

        return ground_truth_deep, ground_truth_mask_deep

class MultipleImageSampler:
    def __init__(self, image_samplers: dict):
        self.image_size = image_samplers[next(iter(image_samplers.keys()))].image_width
        self.image_samplers = image_samplers

    def obtain_random_image_from_tile(self, tile_id: str, augmentation: bool=True, deep_supervision_downsamples=0):
        wsi_id = model_data_manager.data_information.loc[tile_id, "source_wsi"]
        image_cat, ground_truth, ground_truth_mask = self.image_samplers[wsi_id].obtain_random_image_from_tile(tile_id, augmentation=augmentation)
        if deep_supervision_downsamples > 0:
            ground_truth_deep, ground_truth_mask_deep = self.image_samplers[wsi_id].obtain_deep_supervision_outputs(ground_truth, ground_truth_mask, deep_supervision_downsamples=deep_supervision_downsamples)
            return image_cat, ground_truth, ground_truth_mask, ground_truth_deep, ground_truth_mask_deep
        return image_cat, ground_truth, ground_truth_mask

    def obtain_random_sample_batch(self, tile_id: list[str], augmentation: bool=True, deep_supervision_downsamples=0):
        batch_size = len(tile_id)
        image_size = self.image_size
        image_cat_batch = torch.zeros((batch_size, 4, image_size, image_size), dtype=torch.float32, device=config.device)
        image_ground_truth_batch = torch.zeros((batch_size, 1, image_size, image_size), dtype=torch.long, device=config.device)
        image_ground_truth_mask_batch = torch.zeros((batch_size, 1, image_size, image_size), dtype=torch.float32, device=config.device)
        if deep_supervision_downsamples > 0:
            image_ground_truth_deep = []
            image_ground_truth_mask_deep = []
            for k in range(deep_supervision_downsamples):
                scale_factor = 2 ** (k + 1)
                image_ground_truth_deep.append(torch.zeros((batch_size, 1, image_size // scale_factor, image_size // scale_factor), dtype=torch.long, device=config.device))
                image_ground_truth_mask_deep.append(torch.zeros((batch_size, 1, image_size // scale_factor, image_size // scale_factor), dtype=torch.float32, device=config.device))

            for k in range(batch_size):
                image_cat_batch[k, ...], image_ground_truth_batch[k, 0, ...], image_ground_truth_mask_batch[k, 0, ...], image_ground_truth_deep_batch, image_ground_truth_mask_deep_batch =\
                    self.obtain_random_image_from_tile(tile_id[k], augmentation=augmentation, deep_supervision_downsamples=deep_supervision_downsamples)
                for l in range(deep_supervision_downsamples):
                    image_ground_truth_deep[l][k, 0, ...] = image_ground_truth_deep_batch[l]
                    image_ground_truth_mask_deep[l][k, 0, ...] = image_ground_truth_mask_deep_batch[l]
            return image_cat_batch, image_ground_truth_batch, image_ground_truth_mask_batch, image_ground_truth_deep, image_ground_truth_mask_deep
        else:
            for k in range(batch_size):
                image_cat_batch[k, ...], image_ground_truth_batch[k, 0, ...], image_ground_truth_mask_batch[k, 0, ...]=\
                    self.obtain_random_image_from_tile(tile_id[k], augmentation=augmentation, deep_supervision_downsamples=deep_supervision_downsamples)

            return image_cat_batch, image_ground_truth_batch, image_ground_truth_mask_batch


    def obtain_random_sample_with_mixup(self, tile_id1: str, tile_id2: str, mixup_alpha=0.2, augmentation: bool=True, deep_supervision_downsamples=0):
        with torch.no_grad():
            if deep_supervision_downsamples > 0:
                image_cat1, ground_truth1, ground_truth_mask1, ground_truth1_deep, ground_truth_mask1_deep = self.obtain_random_image_from_tile(tile_id1, augmentation=augmentation, deep_supervision_downsamples=deep_supervision_downsamples)
                image_cat2, ground_truth2, ground_truth_mask2, ground_truth2_deep, ground_truth_mask2_deep = self.obtain_random_image_from_tile(tile_id2, augmentation=augmentation, deep_supervision_downsamples=deep_supervision_downsamples)
            else:
                image_cat1, ground_truth1, ground_truth_mask1 = self.obtain_random_image_from_tile(tile_id1, augmentation=augmentation)
                image_cat2, ground_truth2, ground_truth_mask2 = self.obtain_random_image_from_tile(tile_id2, augmentation=augmentation)

            # convert ground truth tensors to one-hot encoding
            ground_truth1 = torch.nn.functional.one_hot(ground_truth1, num_classes=3).permute(2, 0, 1).to(torch.float32)
            ground_truth2 = torch.nn.functional.one_hot(ground_truth2, num_classes=3).permute(2, 0, 1).to(torch.float32)

            gc.collect()
            torch.cuda.empty_cache()

            # apply mixup augmentation
            lambda_interp = np.random.beta(mixup_alpha, mixup_alpha)
            image_cat = lambda_interp * image_cat1 + (1 - lambda_interp) * image_cat2
            ground_truth = lambda_interp * ground_truth1 + (1 - lambda_interp) * ground_truth2
            ground_truth_mask = ground_truth_mask1 * ground_truth_mask2

            del image_cat1, image_cat2, ground_truth1, ground_truth2, ground_truth_mask1, ground_truth_mask2
            gc.collect()
            torch.cuda.empty_cache()

            if deep_supervision_downsamples > 0:
                ground_truth_deep = []
                ground_truth_mask_deep = []
                for k in range(deep_supervision_downsamples):
                    ground_truth_deep.append(
                        lambda_interp * torch.nn.functional.one_hot(ground_truth1_deep[k], num_classes=3).permute(2, 0, 1).to(
                            torch.float32)
                        + (1 - lambda_interp) * torch.nn.functional.one_hot(ground_truth2_deep[k], num_classes=3).permute(2, 0, 1).to(
                            torch.float32))
                    ground_truth_mask_deep.append(ground_truth_mask1_deep[k] * ground_truth_mask2_deep[k])

                return image_cat, ground_truth, ground_truth_mask, ground_truth_deep, ground_truth_mask_deep

            return image_cat, ground_truth, ground_truth_mask

    def obtain_random_sample_with_mixup_batch(self, tile_id1: list[str], tile_id2: list[str], mixup_alpha=0.2, augmentation: bool=True, deep_supervision_downsamples=0):
        batch_size = len(tile_id1)
        image_size = self.image_size
        image_cat_batch = torch.zeros((batch_size, 4, image_size, image_size), dtype=torch.float32, device=config.device)
        image_ground_truth_batch = torch.zeros((batch_size, 3, image_size, image_size), dtype=torch.float32, device=config.device)
        image_ground_truth_mask_batch = torch.zeros((batch_size, 1, image_size, image_size), dtype=torch.float32, device=config.device)

        if deep_supervision_downsamples > 0:
            image_ground_truth_deep = []
            image_ground_truth_mask_deep = []
            for k in range(deep_supervision_downsamples):
                scale_factor = 2 ** (k + 1)
                image_ground_truth_deep.append(
                    torch.zeros((batch_size, 3, image_size // scale_factor, image_size // scale_factor),
                                dtype=torch.float32, device=config.device))
                image_ground_truth_mask_deep.append(
                    torch.zeros((batch_size, 1, image_size // scale_factor, image_size // scale_factor),
                                dtype=torch.float32, device=config.device))

            for k in range(batch_size):
                image_cat_batch[k, ...], image_ground_truth_batch[k, ...], image_ground_truth_mask_batch[k, 0, ...],\
                    image_ground_truth_deep_batch, image_ground_truth_mask_deep_batch = self.obtain_random_sample_with_mixup(
                        tile_id1[k], tile_id2[k], mixup_alpha, augmentation=augmentation, deep_supervision_downsamples=deep_supervision_downsamples)
                for l in range(deep_supervision_downsamples):
                    image_ground_truth_deep[l][k, ...] = image_ground_truth_deep_batch[l]
                    image_ground_truth_mask_deep[l][k, 0, ...] = image_ground_truth_mask_deep_batch[l]

            return image_cat_batch, image_ground_truth_batch, image_ground_truth_mask_batch, image_ground_truth_deep, image_ground_truth_mask_deep
        else:
            for k in range(batch_size):
                image_cat_batch[k, ...], image_ground_truth_batch[k, ...], image_ground_truth_mask_batch[k, 0, ...]=\
                    self.obtain_random_sample_with_mixup(tile_id1[k], tile_id2[k], mixup_alpha, augmentation=augmentation, deep_supervision_downsamples=deep_supervision_downsamples)
            return image_cat_batch, image_ground_truth_batch, image_ground_truth_mask_batch


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
            r.generate_interior_pixels(interior_box=12)
            del r

if not os.path.isfile(os.path.join(folder, "wsi512_1_region.png")):
    with h5py.File(os.path.join(folder, "data.hdf5"), "r") as segmentation_masks:
        for wsi_id in range(1, 3):
            print("Generating images for wsi_{}".format(wsi_id))
            group = segmentation_masks["wsi_{}".format(wsi_id)]
            r = Region(wsi_id, group, writable=False)
            r.load_from_hdf5()
            background = images.get_whole_image(wsi_id)[..., ::-1]
            r.save_region_image(os.path.join(folder, "wsi512_{}_region.png".format(wsi_id)), background)
            r.save_interior_pixels_image(os.path.join(folder, "wsi512_{}_interior.png".format(wsi_id)), background)
            r.save_rotated_interior_pixels_image(os.path.join(folder, "wsi512_{}_rotated.png".format(wsi_id)), background)

all_wsi_masks = h5py.File(os.path.join(folder, "data.hdf5"), "r")
def get_wsi_region_mask(wsi_id: int, use_async=None) -> Region:
    if use_async is not None:
        use_async["image_wsi_sampling"] = {}
        async_files = use_async["image_wsi_sampling"]
        if "all_wsi_masks" not in async_files:
            async_files["all_wsi_masks"] = h5py.File(os.path.join(folder, "data.hdf5"), "r")
        l_all_wsi_masks = async_files["all_wsi_masks"]
    else:
        l_all_wsi_masks = all_wsi_masks

    group = l_all_wsi_masks["wsi_{}".format(wsi_id)]
    r = Region(wsi_id, group, writable=False)
    r.load_from_hdf5()
    return r

def generate_masks_from_subdata(subdata_name: str):
    assert model_data_manager.subdata_exists(subdata_name), "The subdata {} does not exist!".format(subdata_name)
    entries = model_data_manager.get_subdata_entry_list(subdata_name)
    print("Generating masks, looping through unique WSIs......")

    if not os.path.isdir(os.path.join(folder, subdata_name)):
        os.mkdir(os.path.join(folder, subdata_name))
    with h5py.File(os.path.join(folder, subdata_name, "data.hdf5"), "w") as file:
        for wsi_id in tqdm.tqdm(model_data_manager.data_information["source_wsi"].loc[entries].unique()):
            wsi_tiles = model_data_manager.data_information.loc[entries].loc[model_data_manager.data_information.loc[entries, "source_wsi"] == wsi_id].index
            wsi_set = file.create_group("wsi_{}".format(wsi_id))

            r = Region(wsi_id, wsi_set, writable=True)
            r.generate_region(wsi_tiles)
            r.generate_interior_pixels(interior_box=512, wsi_tile_write=wsi_tiles)
            del r

    print("Generating images...")
    with h5py.File(os.path.join(folder, subdata_name, "data.hdf5"), "r") as file:
        for wsi_id in tqdm.tqdm(model_data_manager.data_information["source_wsi"].loc[entries].unique()):
            group = file["wsi_{}".format(wsi_id)]
            r = Region(wsi_id, group, writable=False)
            r.load_from_hdf5()
            background = images.get_whole_image(wsi_id)[..., ::-1]

            r.save_region_image(os.path.join(folder, subdata_name, "wsi512_{}_region.png".format(wsi_id)), background)
            r.save_interior_pixels_image(os.path.join(folder, subdata_name, "wsi512_{}_interior.png".format(wsi_id)), background)
            r.save_rotated_interior_pixels_image(os.path.join(folder, subdata_name, "wsi512_{}_rotated.png".format(wsi_id)), background)

def get_subdata_mask(subdata_name: str):
    assert model_data_manager.subdata_exists(subdata_name), "The subdata {} does not exist!".format(subdata_name)
    assert os.path.isdir(os.path.join(folder, subdata_name)), "The subdata {} has not been created! Use generate_masks_from_subdata".format(subdata_name)
    entries = model_data_manager.get_subdata_entry_list(subdata_name)
    file = h5py.File(os.path.join(folder, subdata_name, "data.hdf5"), "r")
    masks = {}
    for wsi_id in model_data_manager.data_information["source_wsi"].loc[entries].unique():
        masks[wsi_id] = Region(wsi_id, file["wsi_{}".format(wsi_id)], writable=False)
        masks[wsi_id].load_from_hdf5()
    return masks

def get_image_sampler(subdata_name: str, image_width: int, device=config.device, use_async=None) -> MultipleImageSampler:
    """
    If use_async is not None, it should be a dict to store the default h5py files.
    """
    if use_async is not None:
        assert isinstance(use_async, dict), "use_async should be a dict to store the default h5py files."
    mask = get_subdata_mask(subdata_name)

    entries = model_data_manager.get_subdata_entry_list(subdata_name)
    samplers = {}
    for wsi_id in model_data_manager.data_information["source_wsi"].loc[entries].unique():
        sampler = ImageSampler(get_wsi_region_mask(wsi_id, use_async), mask[wsi_id],
                               obtain_reconstructed_binary_segmentation.get_default_WSI_mask(wsi_id, use_async), image_width, device=device)
        samplers[wsi_id] = sampler

    return MultipleImageSampler(samplers)

def generate_image_example(sampler: ImageSampler, tile: str, num: int) -> float:
    ctime = time.time()
    image_comb, ground_truth, ground_truth_mask = sampler.obtain_random_image_from_tile(tile)
    ctime = time.time() - ctime

    image = image_comb[:3, ...].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    region_mask = image_comb[3, ...].detach().cpu().numpy().astype(np.uint8) * 255
    region_mask = np.repeat(np.expand_dims(region_mask, axis=-1), axis=-1, repeats=3)

    ground_truth = ground_truth.detach().cpu().numpy()
    ground_truth_mask = ground_truth_mask.detach().cpu().numpy().astype(np.uint8) * 255
    ground_truth_mask = np.repeat(np.expand_dims(ground_truth_mask, axis=-1), axis=-1, repeats=3)
    classes_image = np.zeros_like(image, dtype=np.uint8)

    # convert ground_truth to hsv
    hue_mask = (255 * ground_truth.astype(np.float32) / 3).astype(np.uint8)
    saturation_mask = ((ground_truth > 0).astype(np.float32) * 255).astype(np.uint8)
    value_mask = saturation_mask

    classes_image[:, :, 0] = hue_mask
    classes_image[:, :, 1] = saturation_mask
    classes_image[:, :, 2] = value_mask

    classes_image = cv2.cvtColor(classes_image, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(image, 0.5, region_mask, 0.5, 0)
    classes_image = cv2.addWeighted(classes_image, 0.5, ground_truth_mask, 0.5, 0)

    # save images
    if not os.path.isdir(os.path.join(folder, "examples")):
        os.mkdir(os.path.join(folder, "examples"))
    cv2.imwrite(os.path.join(folder, "examples", "{}_{}_image.png".format(tile, num)), image)
    cv2.imwrite(os.path.join(folder, "examples", "{}_{}_classes.png".format(tile, num)), classes_image)

    return ctime

if __name__ == "__main__":
    #generate_masks_from_subdata("dataset1_split1")
    #generate_masks_from_subdata("dataset1_split2")
    #generate_masks_from_subdata("dataset1_regional_split1")
    #generate_masks_from_subdata("dataset1_regional_split2")

    mask1 = get_subdata_mask("dataset1_regional_split1")
    sampler = ImageSampler(get_wsi_region_mask(1), mask1[1], obtain_reconstructed_binary_segmentation.get_default_WSI_mask(1), 1024)

    tiles = ["5ac25a1e40dd", "39b8aafd630b", "8e90e6189c6b", "f45a29109ff5"]
    all_time_elapsed = []
    for tile in tiles:
        print("Sampling from tile {}".format(tile))
        for i in tqdm.tqdm(range(10)):
            time_elapsed = generate_image_example(sampler, tile, i)
            time.sleep(0.4)
            all_time_elapsed.append(time_elapsed)

    all_time_elapsed = np.array(all_time_elapsed)
    print("Average time elapsed: {} seconds".format(np.mean(all_time_elapsed)))
    print("Median time elapsed: {} seconds".format(np.median(all_time_elapsed)))
    print("Min time elapsed: {} seconds".format(np.min(all_time_elapsed)))
    print("Max time elapsed: {} seconds".format(np.max(all_time_elapsed)))
    print("First time elapsed: {} seconds".format(all_time_elapsed[0]))
    print("Last time elapsed: {} seconds".format(all_time_elapsed[-1]))