import gc
import os
import shutil
import time
import json

import numpy as np
import pandas as pd
import cv2
import h5py

import config
import model_data_manager
import obtain_reconstructed_wsi_images

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
def get_modified_mask(mask: np.ndarray, top_left_x: int, top_left_y: int):
    larger_mask = np.zeros((mask.shape[0] + 8, mask.shape[1] + 8), dtype=np.uint8)
    larger_mask[4:-4, 4:-4] = mask.astype(np.uint8)

    top_left_x, top_left_y = top_left_x - 4, top_left_y - 4

    inner_mask = cv2.erode(larger_mask, kernel, iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    outer_mask = cv2.dilate(larger_mask, kernel, iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    larger_mask = larger_mask.astype(bool)
    inner_mask = inner_mask.astype(bool)
    outer_mask = outer_mask.astype(bool)

    exists_x = np.argwhere(np.any(outer_mask, axis=0)).squeeze(-1)
    exists_y = np.argwhere(np.any(outer_mask, axis=1)).squeeze(-1)

    bounds_x1, bounds_x2 = np.min(exists_x), np.max(exists_x)
    bounds_y1, bounds_y2 = np.min(exists_y), np.max(exists_y)

    inner_mask = inner_mask[bounds_y1:bounds_y2+1, bounds_x1:bounds_x2+1]
    outer_mask = outer_mask[bounds_y1:bounds_y2+1, bounds_x1:bounds_x2+1]
    mask = larger_mask[bounds_y1:bounds_y2+1, bounds_x1:bounds_x2+1]
    boundary_mask = np.logical_xor(inner_mask, outer_mask)

    top_left_x, top_left_y = top_left_x + bounds_x1, top_left_y + bounds_y1

    return mask, inner_mask, boundary_mask, top_left_x, top_left_y

def assign_mask(combined_mask: np.ndarray, mask: np.ndarray, top_left_x: int, top_left_y: int):
    height, width = combined_mask.shape

    # Compute the shrunk bounds of the mask
    x1, x2 = top_left_x, top_left_x + mask.shape[1]
    y1, y2 = top_left_y, top_left_y + mask.shape[0]

    mask_x1, mask_x2 = 0, mask.shape[1]
    mask_y1, mask_y2 = 0, mask.shape[0]

    if x1 < 0:
        mask_x1 = -x1
        x1 = 0
    if x2 > width:
        mask_x2 = mask.shape[1] - (x2 - width)
        x2 = width
    if y1 < 0:
        mask_y1 = -y1
        y1 = 0
    if y2 > height:
        mask_y2 = mask.shape[0] - (y2 - height)
        y2 = height

    # Assignment of masks
    combined_mask[y1:y2, x1:x2] = np.maximum(combined_mask[y1:y2, x1:x2], mask[mask_y1:mask_y2, mask_x1:mask_x2])

def relative_mask_set(mask: np.ndarray, top_left_x: int, top_left_y: int, new_mask: np.ndarray, new_top_left_x: int, new_top_left_y: int, function: callable=np.maximum):
    """
    Uses given function to write the new_mask onto mask, where out of bounds pixels are ignored.
    :param mask: The mask to be set
    :param top_left_x: The top left x coordinate of the mask
    :param top_left_y: The top left y coordinate of the mask
    :param new_mask: The new mask
    :param new_top_left_x: The top left x coordinate of the new mask
    :param new_top_left_y: The top left y coordinate of the new mask
    :param function: The function to use to set the mask. Default np.maximum
    """

    new_mask_x, new_mask_y = 0, 0
    new_mask_height, new_mask_width = new_mask.shape

    # Shrink now
    if new_top_left_x < top_left_x:
        new_mask_x = top_left_x - new_top_left_x
        new_top_left_x = top_left_x
        new_mask_width -= new_mask_x
    if new_top_left_y < top_left_y:
        new_mask_y = top_left_y - new_top_left_y
        new_top_left_y = top_left_y
        new_mask_height -= new_mask_y
    if new_top_left_x + new_mask_width > top_left_x + mask.shape[1]:
        new_mask_width = top_left_x + mask.shape[1] - new_top_left_x
    if new_top_left_y + new_mask_height > top_left_y + mask.shape[0]:
        new_mask_height = top_left_y + mask.shape[0] - new_top_left_y

    mask[new_top_left_y - top_left_y:new_top_left_y - top_left_y + new_mask_height,
            new_top_left_x - top_left_x:new_top_left_x - top_left_x + new_mask_width] = \
        function(mask[new_top_left_y - top_left_y:new_top_left_y - top_left_y + new_mask_height,
            new_top_left_x - top_left_x:new_top_left_x - top_left_x + new_mask_width],
            new_mask[new_mask_y:new_mask_y + new_mask_height, new_mask_x:new_mask_x + new_mask_width])


class WSIMask:
    def __init__(self, combined_masks: h5py.File, wsi_id: int):
        key = "wsi_{}".format(wsi_id)
        assert key in combined_masks, "wsi_{} does not exist in the combined_masks.hdf5 file".format(wsi_id)
        self.wsi_whole_masks = combined_masks[key]

    def obtain_unknown_mask(self, x1: int, x2: int, y1: int, y2: int):
        return np.array(self.wsi_whole_masks["unknown"][y1:y2, x1:x2], dtype=np.uint8)

    def obtain_blood_vessel_mask(self, x1: int, x2: int, y1: int, y2: int):
        return np.array(self.wsi_whole_masks["blood_vessel"][y1:y2, x1:x2], dtype=np.uint8)

    def obtain_glomerulus_mask(self, x1: int, x2: int, y1: int, y2: int):
        return np.array(self.wsi_whole_masks["glomerulus"][y1:y2, x1:x2], dtype=np.uint8)


if __name__ == '__main__':
    if not os.path.isfile(os.path.join("segmentation_reconstructed_data", "polygons_reconstructed.hdf5")):
        print("polygons_reconstructed.hdf5 does not exist in the segmentation_reconstructed_data folder. Please run obtain_reconstructed_polygons.py to obtain the reconstructed polygons first.")
        quit(-1)

    if os.path.isfile(os.path.join("segmentation_reconstructed_data", "segmentation_masks.hdf5")):
        os.remove(os.path.join("segmentation_reconstructed_data", "segmentation_masks.hdf5"))

    if os.path.isfile(os.path.join("segmentation_reconstructed_data", "combined_masks.hdf5")):
        os.remove(os.path.join("segmentation_reconstructed_data", "combined_masks.hdf5"))

    # load the reconstructed polygons, and compute the boundaries and interiors
    print("Computing polygons boundaries and interiors....")
    with h5py.File(os.path.join("segmentation_reconstructed_data", "polygons_reconstructed.hdf5"), "r") as reconstructed_polygons:
        with h5py.File(os.path.join("segmentation_reconstructed_data", "segmentation_masks.hdf5"), "w") as segmentation_masks:
            for wsi_id in range(1, 15):
                if wsi_id != 5:
                    wsi_group = "wsi_{}".format(wsi_id)
                    polygons_group = reconstructed_polygons[wsi_group]
                    modified_polygons_group = segmentation_masks.create_group(wsi_group)

                    num_polygons = polygons_group["num_polygons"][()]
                    modified_polygons_group.create_dataset("num_polygons", data=num_polygons, dtype=np.int32)

                    for polygon_id in range(num_polygons):
                        polygon = polygons_group[str(polygon_id)]
                        polygon_type = polygon["type"].asstr()[()]
                        polygon_top_left_x = polygon["top_left_x"][()]
                        polygon_top_left_y = polygon["top_left_y"][()]
                        polygon_mask = np.array(polygon["mask"])

                        mask, inner_mask, boundary_mask, top_left_x, top_left_y = get_modified_mask(polygon_mask, polygon_top_left_x, polygon_top_left_y)

                        modified_polygon = modified_polygons_group.create_group(str(polygon_id))
                        modified_polygon.create_dataset("type", data=polygon_type, dtype=h5py.string_dtype())
                        modified_polygon.create_dataset("top_left_x", data=top_left_x, dtype=np.int32)
                        modified_polygon.create_dataset("top_left_y", data=top_left_y, dtype=np.int32)
                        modified_polygon.create_dataset("mask", data=mask, dtype=bool, shape=mask.shape, compression="gzip", compression_opts=9)
                        modified_polygon.create_dataset("inner_mask", data=inner_mask, dtype=bool, shape=inner_mask.shape, compression="gzip", compression_opts=9)
                        modified_polygon.create_dataset("boundary_mask", data=boundary_mask, dtype=bool, shape=boundary_mask.shape, compression="gzip", compression_opts=9)

    print("Computing global wsi masks....")
    # now load the computed polygons, and compute the per wsi mask
    with h5py.File(os.path.join("segmentation_reconstructed_data", "segmentation_masks.hdf5"), "r") as segmentation_masks:
        with h5py.File(os.path.join("segmentation_reconstructed_data", "combined_masks.hdf5"), "w") as combined_masks:
            for wsi_id in range(1, 15):
                if wsi_id != 5:
                    wsi_group = "wsi_{}".format(wsi_id)
                    wsi_tiles = model_data_manager.data_information.loc[model_data_manager.data_information["source_wsi"] == wsi_id]
                    modified_polygons_group = segmentation_masks[wsi_group]

                    wsi_information = model_data_manager.data_information
                    wsi_information = wsi_information.loc[wsi_information["source_wsi"] == wsi_id]
                    width = int(wsi_information["i"].max() + 512)
                    height = int(wsi_information["j"].max() + 512)

                    combined_mask_unknown = np.zeros((height, width), dtype=np.uint8)
                    combined_mask_glomerulus = np.zeros((height, width), dtype=np.uint8)
                    combined_mask_blood_vessel = np.zeros((height, width), dtype=np.uint8)

                    num_polygons = modified_polygons_group["num_polygons"][()]

                    for polygon_id in range(num_polygons):
                        polygon = modified_polygons_group[str(polygon_id)]
                        polygon_type = polygon["type"].asstr()[()]
                        polygon_top_left_x = polygon["top_left_x"][()]
                        polygon_top_left_y = polygon["top_left_y"][()]
                        polygon_mask = np.array(polygon["inner_mask"], dtype=np.uint8)
                        polygon_boundary_mask = np.array(polygon["boundary_mask"], dtype=np.uint8) * 2

                        # Fix the boundary problem of dataset 2.
                        boundary_clearance = 16 # how many boundary pixels to ignore
                        if polygon_type == "blood_vessel":
                            neighborhood_tiles = wsi_tiles.loc[(polygon_top_left_x - boundary_clearance - 512 < wsi_tiles["i"]) &
                                            (wsi_tiles["i"] < polygon_top_left_x + boundary_clearance + polygon_mask.shape[1]) &
                                            (polygon_top_left_y - boundary_clearance - 512 < wsi_tiles["j"]) &
                                            (wsi_tiles["j"] < polygon_top_left_y + boundary_clearance + polygon_mask.shape[0]) &
                                            (wsi_tiles["dataset"] != 1)].index

                            clearance_top_left_x = polygon_top_left_x - boundary_clearance
                            clearance_top_left_y = polygon_top_left_y - boundary_clearance
                            clearance_mask = np.zeros((polygon_mask.shape[0] + boundary_clearance * 2,
                                                       polygon_mask.shape[1] + boundary_clearance * 2), dtype=np.uint8)

                            central_mask = np.zeros((512 + 2 * boundary_clearance, 512 + 2 * boundary_clearance), dtype=np.uint8)
                            small_central_mask = np.ones((512 + 2 * boundary_clearance, 512 + 2 * boundary_clearance), dtype=np.uint8)
                            central_mask[boundary_clearance:-boundary_clearance, boundary_clearance:-boundary_clearance] = 1
                            small_central_mask[2 * boundary_clearance:-2 * boundary_clearance, 2 * boundary_clearance:-2 * boundary_clearance] = 0

                            # Compute the pixels we have to ignore
                            for tile_id in neighborhood_tiles:
                                neighborhood_problem_mask = np.zeros((512 + 2 * boundary_clearance, 512 + 2 * boundary_clearance), dtype=np.uint8)
                                relative_mask_set(neighborhood_problem_mask, wsi_tiles.loc[tile_id]["i"] - boundary_clearance,
                                                  wsi_tiles.loc[tile_id]["j"] - boundary_clearance, polygon_boundary_mask // 2, polygon_top_left_x, polygon_top_left_y)
                                neighborhood_expanded_problem_mask = cv2.dilate(
                                    neighborhood_problem_mask * small_central_mask,
                                    kernel, iterations=boundary_clearance, borderType=cv2.BORDER_CONSTANT, borderValue=0)

                                neighborhood_problem_mask = (neighborhood_problem_mask * (1 - central_mask) +\
                                                            neighborhood_expanded_problem_mask * central_mask) * small_central_mask

                                relative_mask_set(clearance_mask, clearance_top_left_x, clearance_top_left_y, neighborhood_problem_mask,
                                                  wsi_tiles.loc[tile_id]["i"] - boundary_clearance, wsi_tiles.loc[tile_id]["j"] - boundary_clearance)

                                del neighborhood_problem_mask, neighborhood_expanded_problem_mask
                            del central_mask
                            gc.collect()

                            relative_mask_set(clearance_mask, clearance_top_left_x, clearance_top_left_y, 1 - polygon_mask, polygon_top_left_x, polygon_top_left_y,
                                              np.minimum)
                            relative_mask_set(polygon_boundary_mask, polygon_top_left_x, polygon_top_left_y, 1 - clearance_mask, clearance_top_left_x, clearance_top_left_y,
                                              np.minimum)
                            assign_mask(combined_mask_unknown, clearance_mask, clearance_top_left_x, clearance_top_left_y)


                        # Assignment of masks
                        if polygon_type == "unsure":
                            assign_mask(combined_mask_unknown, np.maximum(polygon_mask, polygon_boundary_mask), polygon_top_left_x, polygon_top_left_y)
                        elif polygon_type == "glomerulus":
                            assign_mask(combined_mask_glomerulus, np.maximum(polygon_mask, polygon_boundary_mask), polygon_top_left_x, polygon_top_left_y)
                        elif polygon_type == "blood_vessel":
                            assign_mask(combined_mask_blood_vessel, np.maximum(polygon_mask, polygon_boundary_mask), polygon_top_left_x, polygon_top_left_y)
                        else:
                            raise ValueError("Unknown polygon type: {}".format(polygon_type))

                    combined_wsi_masks = combined_masks.create_group(wsi_group)
                    combined_wsi_masks.create_dataset("unknown", data=combined_mask_unknown, dtype=np.uint8, compression="gzip", compression_opts=9, shape=combined_mask_unknown.shape, chunks=(512, 512))
                    combined_wsi_masks.create_dataset("glomerulus", data=combined_mask_glomerulus, dtype=np.uint8, compression="gzip", compression_opts=9, shape=combined_mask_glomerulus.shape, chunks=(512, 512))
                    combined_wsi_masks.create_dataset("blood_vessel", data=combined_mask_blood_vessel, dtype=np.uint8, compression="gzip", compression_opts=9, shape=combined_mask_blood_vessel.shape, chunks=(512, 512))

    gc.collect()
    # finally, we render the images
    print("Rendering images....")
    with h5py.File(os.path.join("segmentation_reconstructed_data", "combined_masks.hdf5"), "r") as combined_masks:
        for wsi_id in range(1, 15):
            if wsi_id != 5:
                print("Rendering image for WSI {}...".format(wsi_id))
                gc.collect()
                wsi_group = "wsi_{}".format(wsi_id)
                combined_wsi_masks = combined_masks[wsi_group]

                combined_mask_unknown = np.array(combined_wsi_masks["unknown"], dtype=np.uint8)
                combined_mask_glomerulus = np.array(combined_wsi_masks["glomerulus"], dtype=np.uint8)
                combined_mask_blood_vessel = np.array(combined_wsi_masks["blood_vessel"], dtype=np.uint8)

                image_np = obtain_reconstructed_wsi_images.construct_wsi(wsi_id) # (height, width, 3) RGB image.

                image1, image2 = np.copy(image_np), np.copy(image_np)

                image1[(combined_mask_unknown == 1) | (combined_mask_glomerulus == 1) | (combined_mask_blood_vessel == 1), :] = 0
                image2[(combined_mask_unknown == 2) | (combined_mask_glomerulus == 2) | (combined_mask_blood_vessel == 2), :] = 0

                image1[combined_mask_unknown == 1, 2] = 255
                image2[combined_mask_unknown == 2, 2] = 255
                image1[combined_mask_glomerulus == 1, 1] = 255
                image2[combined_mask_glomerulus == 2, 1] = 255
                image1[combined_mask_blood_vessel == 1, 0] = 255
                image2[combined_mask_blood_vessel == 2, 0] = 255

                image_np = (image_np.astype(np.float32) / 10 + (3 * image1.astype(np.float32)) / 10 + (6 * image2.astype(np.float32)) / 10).astype(np.uint8)

                # save the image
                image_np = image_np[:, :, ::-1].copy() # rgb to bgr
                cv2.imwrite(os.path.join("segmentation_reconstructed_data", "wsi_{}.png".format(wsi_id)), image_np)

                # save the image with grids
                obtain_reconstructed_wsi_images.draw_grids(image_np)
                cv2.imwrite(os.path.join("segmentation_reconstructed_data", "wsi_{}_grids.png".format(wsi_id)), image_np)
else:
    if not os.path.isfile(os.path.join("segmentation_reconstructed_data", "combined_masks.hdf5")):
        raise ValueError("Cannot find combined_masks.hdf5 in segmentation_reconstructed_data folder. You should run obtain_reconstructed_wsi_masks.py first.")

    default_masks_file = h5py.File(os.path.join("segmentation_reconstructed_data", "combined_masks.hdf5"), "r")
    def get_default_WSI_mask(wsi_id: int, use_async=None) -> WSIMask:
        if use_async is not None:
            use_async["obtain_reconstructed_binary_segmentation"] = {}
            async_files = use_async["obtain_reconstructed_binary_segmentation"]
            if "default_masks_file" not in async_files:
                async_files["default_masks_file"] = h5py.File(os.path.join("segmentation_reconstructed_data", "combined_masks.hdf5"), "r")
            l_default_masks_file = async_files["default_masks_file"]
        else:
            l_default_masks_file = default_masks_file
        return WSIMask(l_default_masks_file, wsi_id)