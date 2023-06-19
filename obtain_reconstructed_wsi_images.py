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


def has_neighbors(wsi_tile):
    tile_info = model_data_manager.data_information
    wsi_x = tile_info.loc[wsi_tile, "i"]
    wsi_y = tile_info.loc[wsi_tile, "j"]
    dataset = tile_info.loc[wsi_tile, "dataset"]

    same_dataset_tile_info = tile_info.loc[tile_info["dataset"] == dataset]

    existing_data = np.zeros(shape=(1024, 1024), dtype=bool)
    existing_data[256:768, 256:768] = True

    # We check whether the whole neighboring region can be filled. This means existing data must be True in all pixels.

    # Use a while loop. In each iteration, select a element in existing_data that is False.
    while True:
        if np.all(existing_data):
            return True

        false_element = np.argwhere(existing_data == False)[0, :]
        false_element_x = false_element[0] - 256 + wsi_x
        false_element_y = false_element[1] - 256 + wsi_y

        # Check whether there exist a row in same_dataset_tile_info such that the i, j columns satisfy:
        # false_element_x in [i, i + 512) and false_element_y in [j, j + 512)
        # If such a row exists, then we can fill the region.

        pcols = (same_dataset_tile_info["i"] <= false_element_x) & (false_element_x < same_dataset_tile_info["i"] + 512) & \
            (same_dataset_tile_info["j"] <= false_element_y) & (false_element_y < same_dataset_tile_info["j"] + 512)

        if pcols.any():
            pcols = same_dataset_tile_info.loc[pcols]
            for neighboring_wsi in pcols.index:
                neighboring_wsi_x = same_dataset_tile_info.loc[neighboring_wsi, "i"]
                neighboring_wsi_y = same_dataset_tile_info.loc[neighboring_wsi, "j"]

                neighboring_wsi_x1 = neighboring_wsi_x - wsi_x + 256
                neighboring_wsi_y1 = neighboring_wsi_y - wsi_y + 256

                neighboring_wsi_x2 = neighboring_wsi_x1 + 512
                neighboring_wsi_y2 = neighboring_wsi_y1 + 512

                neighboring_wsi_x1 = max(neighboring_wsi_x1, 0)
                neighboring_wsi_y1 = max(neighboring_wsi_y1, 0)
                neighboring_wsi_x2 = min(neighboring_wsi_x2, 1024)
                neighboring_wsi_y2 = min(neighboring_wsi_y2, 1024)

                existing_data[neighboring_wsi_x1:neighboring_wsi_x2, neighboring_wsi_y1:neighboring_wsi_y2] = True
        else:
            return False

def construct_wsi(wsi_id):
    wsi_information = model_data_manager.data_information
    wsi_information = wsi_information.loc[wsi_information["source_wsi"] == wsi_id]
    width = int(wsi_information["i"].max() + 512)
    height = int(wsi_information["j"].max() + 512)

    image_np = np.zeros((height, width, 3), dtype=np.uint8)

    for wsi_tile in wsi_information.index:
        x = wsi_information.loc[wsi_tile, "i"]
        y = wsi_information.loc[wsi_tile, "j"]
        image_np[y:y+512, x:x+512, :] = image_loader.get_image_data(wsi_tile)

    return image_np

def construct_wsi_with_polygons(wsi_id):
    wsi_information = model_data_manager.data_information
    wsi_information = wsi_information.loc[wsi_information["source_wsi"] == wsi_id]
    width = int(wsi_information["i"].max() + 512)
    height = int(wsi_information["j"].max() + 512)

    image_np = np.zeros((height, width, 3), dtype=np.uint8)

    for wsi_tile in wsi_information.index:
        x = wsi_information.loc[wsi_tile, "i"]
        y = wsi_information.loc[wsi_tile, "j"]
        wsi_tile_img = image_loader.get_image_data(wsi_tile)

        if wsi_information.loc[wsi_tile, "dataset"] != 1:
            wsi_tile_img //= 2


        if wsi_tile in all_polygon_masks:
            for polygon_mask in all_polygon_masks[wsi_tile]:
                # The color depends on the type, default unknown color = blue
                color = (0, 0, 255)
                if polygon_mask["type"] == "glomerulus":
                    color = (0, 255, 0)  # green
                elif polygon_mask["type"] == "blood_vessel":
                    color = (255, 0, 0)  # red

                # Draw the polygon
                polygon_coordinate_list = polygon_mask["coordinates"][0]  # This is a list of integer 2-tuples, representing the coordinates.
                for i in range(len(polygon_coordinate_list)):
                    cv2.line(wsi_tile_img, polygon_coordinate_list[i],
                             polygon_coordinate_list[(i + 1) % len(polygon_coordinate_list)], color, 3)

                # Fill the polygon with the color, with 35% opacity
                overlay = wsi_tile_img.copy()
                cv2.fillPoly(overlay, [np.array(polygon_coordinate_list)], color)
                wsi_tile_img = cv2.addWeighted(overlay, 0.35, wsi_tile_img, 0.65, 0)

        image_np[y:y + 512, x:x + 512, :] = wsi_tile_img

    return image_np

def draw_grids(image_np):
    for i in range(0, image_np.shape[1], 512):
        cv2.line(image_np, (i, 0), (i, image_np.shape[0]), (0, 255, 255), 1)
    for j in range(0, image_np.shape[0], 512):
        cv2.line(image_np, (0, j), (image_np.shape[1], j), (0, 255, 255), 1)

    return image_np

if __name__ == "__main__":
    if not os.path.exists("reconstructed_wsi_images"):
        os.mkdir("reconstructed_wsi_images")

    # reconstruct WSI images.
    for wsi_id in range(1, 15):
        if wsi_id != 5:
            image_np = construct_wsi(wsi_id)
            cv2.imwrite("reconstructed_wsi_images/wsi_{}.png".format(wsi_id), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            draw_grids(image_np)
            cv2.imwrite("reconstructed_wsi_images/wsi_{}_grids.png".format(wsi_id), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            image_np_with_polygons = construct_wsi_with_polygons(wsi_id)
            cv2.putText(image_np_with_polygons, "Unknown", (512, 512), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 2)
            cv2.putText(image_np_with_polygons, "Glomerulus", (512, 1024), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 2)
            cv2.putText(image_np_with_polygons, "Blood vessel", (512, 1536), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 2)

            cv2.imwrite("reconstructed_wsi_images/wsi_{}_with_polygons.png".format(wsi_id), cv2.cvtColor(image_np_with_polygons, cv2.COLOR_RGB2BGR))

            draw_grids(image_np_with_polygons)
            cv2.imwrite("reconstructed_wsi_images/wsi_{}_with_polygons_grids.png".format(wsi_id), cv2.cvtColor(image_np_with_polygons, cv2.COLOR_RGB2BGR))

            print("WSI {} done".format(wsi_id))

