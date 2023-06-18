"""Displays the intersection of the polygons on the WSIs. Uses the refined polygons. Run obtain_refined_polygons.py first."""

import os
import json

import numpy as np
import cv2

import model_data_manager
import config

if not os.path.isfile(os.path.join(config.input_data_path, "polygons_refined.jsonl")):
    print("polygons_refined.jsonl does not exist in the data folder. Please run obtain_refined_polygons.py to remove the repeated polygons first.")
    quit(-1)


def construct_overlap_info(wsi_id):
    wsi_information = model_data_manager.data_information
    wsi_information = wsi_information.loc[wsi_information["source_wsi"] == wsi_id]
    width = int(wsi_information["i"].max() + 512)
    height = int(wsi_information["j"].max() + 512)

    image_np = np.zeros((height, width, 3), dtype=np.uint8)

    for wsi_tile in wsi_information.index:
        x = wsi_information.loc[wsi_tile, "i"]
        y = wsi_information.loc[wsi_tile, "j"]

        image_np[y:y + 512, x:x + 512, :] += 1

        if wsi_id == 2:
            assert x % 512 == 0, "x is not a multiple of 512. Problematic WSI tile: {}".format(wsi_tile)
            if wsi_information.loc[wsi_tile, "dataset"] == 1:
                assert y % 512 == 0, "y is not a multiple of 512. Problematic WSI tile: {}".format(wsi_tile)
            else:
                if y % 512 != 36:
                    print("y is not a multiple of 512 mod 36. Problematic WSI tile: {}, modulo: {}, x_grid: {}, y_grid: {}".format(wsi_tile, y % 512, x // 512, y // 512))

    assert np.all(image_np <= 2), "Expected at most 2 overlapping regions. Problematic WSI ID: {}".format(wsi_id)
    if wsi_id != 2:
        assert np.all(image_np <= 1), "Expected at most 1 overlapping region. Problematic WSI ID: {}".format(wsi_id)

    return image_np * 64

def draw_grids(image_np):
    for i in range(0, image_np.shape[1], 512):
        cv2.line(image_np, (i, 0), (i, image_np.shape[0]), (0, 255, 255), 3)
    for j in range(0, image_np.shape[0], 512):
        cv2.line(image_np, (0, j), (image_np.shape[1], j), (0, 255, 255), 3)

    return image_np

if not os.path.exists("reconstructed_wsi_images"):
    os.mkdir("reconstructed_wsi_images")

# reconstruct WSI images.
for wsi_id in range(1, 15):
    if wsi_id != 5:
        image_np_with_polygons = construct_overlap_info(wsi_id)
        cv2.putText(image_np_with_polygons, "One overlap", (512, 512), cv2.FONT_HERSHEY_SIMPLEX, 10, (64, 64, 64), 10)
        cv2.putText(image_np_with_polygons, "Two overlap", (512, 1024), cv2.FONT_HERSHEY_SIMPLEX, 10, (128, 128, 128), 10)
        cv2.putText(image_np_with_polygons, "Three overlap", (512, 1536), cv2.FONT_HERSHEY_SIMPLEX, 10, (192, 192, 192), 10)

        cv2.imwrite("reconstructed_wsi_images/wsi_{}_overlap.png".format(wsi_id), cv2.cvtColor(image_np_with_polygons, cv2.COLOR_RGB2BGR))

        draw_grids(image_np_with_polygons)
        cv2.imwrite("reconstructed_wsi_images/wsi_{}_overlap_grids.png".format(wsi_id), cv2.cvtColor(image_np_with_polygons, cv2.COLOR_RGB2BGR))

        print("WSI {} done".format(wsi_id))

