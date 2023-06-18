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

image_loader = model_data_manager.get_dataset_dataloader(None)

with open(os.path.join(config.input_data_path, "polygons_refined.jsonl")) as json_file:
    json_list = list(json_file)

all_polygon_masks = {}
for json_str in json_list:
    polygon_masks = json.loads(json_str)
    all_polygon_masks[polygon_masks["id"]] = polygon_masks["annotations"]

def construct_wsi_with_polygons_intersection(wsi_id, use_whole_intersection=False):
    wsi_information = model_data_manager.data_information
    wsi_information = wsi_information.loc[wsi_information["source_wsi"] == wsi_id]
    width = int(wsi_information["i"].max() + 512)
    height = int(wsi_information["j"].max() + 512)

    image_np = np.zeros((height, width, 3), dtype=np.uint8)

    for wsi_tile in wsi_information.index:
        x = wsi_information.loc[wsi_tile, "i"]
        y = wsi_information.loc[wsi_tile, "j"]
        wsi_tile_img = image_loader.get_image_data(wsi_tile)

        all_history = np.zeros_like(wsi_tile_img)
        overlaps = np.zeros_like(wsi_tile_img)

        if wsi_tile in all_polygon_masks:
            for polygon_mask in all_polygon_masks[wsi_tile]:
                # Draw the polygon
                polygon_coordinate_list = polygon_mask["coordinates"][0]  # This is a list of integer 2-tuples, representing the coordinates.

                polygon_mask = np.zeros_like(wsi_tile_img)
                cv2.fillPoly(polygon_mask, [np.array(polygon_coordinate_list)], (1, 1, 1))

                overlap = np.bitwise_and(polygon_mask, all_history)

                if use_whole_intersection:
                    if np.any(overlap) and np.all(overlap == polygon_mask):
                        overlaps = np.bitwise_or(overlaps, overlap)
                else:
                    if np.any(overlap):
                        overlaps = np.bitwise_or(overlaps, overlap)

                all_history = np.bitwise_or(all_history, polygon_mask)

        image_np[y:y + 512, x:x + 512, :] = overlaps * 255

    return image_np

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
        image_np_with_polygons = construct_wsi_with_polygons_intersection(wsi_id)
        cv2.putText(image_np_with_polygons, "Unknown", (512, 512), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 2)
        cv2.putText(image_np_with_polygons, "Glomerulus", (512, 1024), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 2)
        cv2.putText(image_np_with_polygons, "Blood vessel", (512, 1536), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 2)

        cv2.imwrite("reconstructed_wsi_images/wsi_{}_intersection_refined.png".format(wsi_id), cv2.cvtColor(image_np_with_polygons, cv2.COLOR_RGB2BGR))

        draw_grids(image_np_with_polygons)
        cv2.imwrite("reconstructed_wsi_images/wsi_{}_intersection_refined_grids.png".format(wsi_id), cv2.cvtColor(image_np_with_polygons, cv2.COLOR_RGB2BGR))

        print("WSI {} done".format(wsi_id))

