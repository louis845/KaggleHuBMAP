"""Removes the repeated polygons."""

import os
import json

import numpy as np
import cv2

import model_data_manager
import config

image_loader = model_data_manager.get_dataset_dataloader(None)

with open(os.path.join(config.input_data_path, "polygons.jsonl")) as json_file:
    json_list = list(json_file)


all_polygon_masks = {}

with open(os.path.join(config.input_data_path, "polygons_refined.jsonl"), "w") as json_file:
    for json_str in json_list:
        polygon_masks = json.loads(json_str)
        annotations = polygon_masks["annotations"]

        index = 0
        while index < len(annotations):
            annotation = annotations[index]
            has_equal = False
            for k in range(index + 1, len(annotations)):
                if annotations[k] == annotation:
                    has_equal = True
                    break
            if has_equal:
                annotations.pop(index)
            else:
                index += 1

        refined_masks_str = json.dumps(polygon_masks, separators=(",", ":"))

        json_file.write(refined_masks_str + "\n")