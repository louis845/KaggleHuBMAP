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

                        x1, x2 = polygon_top_left_x, polygon_top_left_x + polygon_mask.shape[1]
                        y1, y2 = polygon_top_left_y, polygon_top_left_y + polygon_mask.shape[0]

                        mask_x1, mask_x2 = 0, polygon_mask.shape[1]
                        mask_y1, mask_y2 = 0, polygon_mask.shape[0]

                        if x1 < 0:
                            mask_x1 = -x1
                            x1 = 0
                        if x2 > width:
                            mask_x2 = polygon_mask.shape[1] - (x2 - width)
                            x2 = width
                        if y1 < 0:
                            mask_y1 = -y1
                            y1 = 0
                        if y2 > height:
                            mask_y2 = polygon_mask.shape[0] - (y2 - height)
                            y2 = height

                        if polygon_type == "unsure":
                            combined_mask_unknown[y1:y2, x1:x2] = np.maximum(combined_mask_unknown[y1:y2, x1:x2], polygon_mask[mask_y1:mask_y2, mask_x1:mask_x2])
                            combined_mask_unknown[y1:y2, x1:x2] = np.maximum(combined_mask_unknown[y1:y2, x1:x2], polygon_boundary_mask[mask_y1:mask_y2, mask_x1:mask_x2])
                        elif polygon_type == "glomerulus":
                            combined_mask_glomerulus[y1:y2, x1:x2] = np.maximum(combined_mask_glomerulus[y1:y2, x1:x2], polygon_mask[mask_y1:mask_y2, mask_x1:mask_x2])
                            combined_mask_glomerulus[y1:y2, x1:x2] = np.maximum(combined_mask_glomerulus[y1:y2, x1:x2], polygon_boundary_mask[mask_y1:mask_y2, mask_x1:mask_x2])
                        elif polygon_type == "blood_vessel":
                            combined_mask_blood_vessel[y1:y2, x1:x2] = np.maximum(combined_mask_blood_vessel[y1:y2, x1:x2], polygon_mask[mask_y1:mask_y2, mask_x1:mask_x2])
                            combined_mask_blood_vessel[y1:y2, x1:x2] = np.maximum(combined_mask_blood_vessel[y1:y2, x1:x2], polygon_boundary_mask[mask_y1:mask_y2, mask_x1:mask_x2])
                        else:
                            raise ValueError("Unknown polygon type: {}".format(polygon_type))

                    combined_wsi_masks = combined_masks.create_group(wsi_group)
                    combined_wsi_masks.create_dataset("unknown", data=combined_mask_unknown, dtype=np.uint8, compression="gzip", compression_opts=9, shape=combined_mask_unknown.shape)
                    combined_wsi_masks.create_dataset("glomerulus", data=combined_mask_glomerulus, dtype=np.uint8, compression="gzip", compression_opts=9, shape=combined_mask_glomerulus.shape)
                    combined_wsi_masks.create_dataset("blood_vessel", data=combined_mask_blood_vessel, dtype=np.uint8, compression="gzip", compression_opts=9, shape=combined_mask_blood_vessel.shape)

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