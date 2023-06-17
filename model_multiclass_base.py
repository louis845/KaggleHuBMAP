import os
import json

import model_data_manager

import numpy as np

def generate_multiclass_config(output_file):
    class_file = {
        "classes": ["blood_vessel"],
        "class_weights": [1.0]
    }
    with open(output_file, "w") as f:
        json.dump(class_file, f, indent=4)

def load_multiclass_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    classes = config["classes"]
    if "class_weights" in config:
        class_weights = config["class_weights"]
    else:
        class_weights = [1.0] * len(classes)
    return config, classes, len(classes), class_weights

def save_multiclass_config(config_file, config):
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

def precompute_classes(dataset_loader: model_data_manager.DatasetDataLoader, total_entries, classes):
    entries_masks = {}
    for entry in total_entries:
        entries_masks[entry] = np.zeros(shape=(512, 512), dtype=np.int64)
        for k in range(len(classes)):
            seg_class = classes[k]
            seg_mask = dataset_loader.get_segmentation_mask(entry, seg_class)
            entries_masks[entry][seg_mask] = k + 1
    return entries_masks