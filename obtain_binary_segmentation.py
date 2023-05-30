import os
import shutil
import time
import json

import numpy as np
import pandas as pd
import cv2
import h5py

import config


if __name__ == '__main__':
    data_information = pd.read_csv(os.path.join(config.input_data_path, "tile_meta.csv"), index_col=0)
    with open(os.path.join(config.input_data_path, "polygons.jsonl")) as json_file:
        json_list = list(json_file)

    all_polygon_masks = {}
    for json_str in json_list:
        polygon_masks = json.loads(json_str)
        all_polygon_masks[polygon_masks["id"]] = polygon_masks["annotations"]

    if not os.path.exists("segmentation_data"):
        os.mkdir("segmentation_data")
    else:
        shutil.rmtree("segmentation_data")
        os.mkdir("segmentation_data")

    num_glomerulus = 0
    num_blood_vessel = 0
    num_unknown = 0

    glomerulus_size = []
    blood_vessel_size = []
    unknown_size = []

    data_summary = h5py.File(os.path.join("segmentation_data", "data_summary.h5"), "w")
    segmentation_data_h5 = data_summary.create_group("segmentation_data")

    print("Starting to compute the binary segmentation. Length: {}".format(len(data_information)))
    count = 0
    ctime = time.time()
    # Compute the binary segmentation here
    for entry in data_information.index:
        image = cv2.imread(os.path.join(config.input_data_path, "train", entry + ".tif"))
        if image.shape != (512, 512, 3):
            print("Image {} has shape {}".format(entry, image.shape))

        # Add the polygon annotations if it exists
        if entry in all_polygon_masks:
            glomerulus_mask = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            blood_vessel_mask = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            unknown_mask = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)

            for polygon_mask in all_polygon_masks[entry]:
                # The color depends on the type, default unknown color = blue
                polygon_coordinate_list = polygon_mask["coordinates"][0]  # This is a list of integer 2-tuples, representing the coordinates.
                mask = np.zeros_like(glomerulus_mask, dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(polygon_coordinate_list)], (1,))
                if polygon_mask["type"] == "glomerulus":
                    glomerulus_mask = np.bitwise_or(glomerulus_mask, mask)
                    num_glomerulus += 1
                    glomerulus_size.append(np.sum(mask))
                elif polygon_mask["type"] == "blood_vessel":
                    blood_vessel_mask = np.bitwise_or(blood_vessel_mask, mask)
                    num_blood_vessel += 1
                    blood_vessel_size.append(np.sum(mask))
                else:
                    unknown_mask = np.bitwise_or(unknown_mask, mask)
                    num_unknown += 1
                    unknown_size.append(np.sum(mask))

            if os.path.exists(os.path.join("segmentation_data", entry)):
                shutil.rmtree(os.path.join("segmentation_data", entry))
                os.mkdir(os.path.join("segmentation_data", entry))
            else:
                os.mkdir(os.path.join("segmentation_data", entry))

            # Save the masks
            # First save them as a npz file
            np.savez_compressed(os.path.join("segmentation_data", entry, "masks.npz"),
                                glomerulus=np.squeeze(glomerulus_mask, axis=-1) > 0,
                                blood_vessel=np.squeeze(blood_vessel_mask, axis=-1) > 0,
                                unknown=np.squeeze(unknown_mask, axis=-1) > 0)

            entry_h5 = segmentation_data_h5.create_group(entry)

            entry_h5.create_dataset(name="glomerulus", data = (np.squeeze(glomerulus_mask, axis=-1) > 0))
            entry_h5.create_dataset(name="blood_vessel", data = (np.squeeze(blood_vessel_mask, axis=-1) > 0))
            entry_h5.create_dataset(name="unknown", data = (np.squeeze(unknown_mask, axis=-1) > 0))

            # Save the masks as 3 png files using cv2.imwrite
            cv2.imwrite(os.path.join("segmentation_data", entry, "glomerulus.png"), glomerulus_mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join("segmentation_data", entry, "blood_vessel.png"), blood_vessel_mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join("segmentation_data", entry, "unknown.png"), unknown_mask.astype(np.uint8) * 255)

        count += 1
        if count % 1000 == 0:
            print("Processed {} images. Time elapsed: {}".format(count, time.time() - ctime))
            ctime = time.time()

    # Save the statistics. Use np sort to sort glomerulus_size, blood_vessel_size, unknown_size, and save them into a statistics.npz file
    np.savez_compressed(os.path.join("segmentation_data", "statistics.npz"),
                        glomerulus_size=np.sort(glomerulus_size),
                        blood_vessel_size=np.sort(blood_vessel_size),
                        unknown_size=np.sort(unknown_size))

    statistics = data_summary.create_group("statistics")
    statistics.create_dataset(name="glomerulus_size", data=np.sort(glomerulus_size))
    statistics.create_dataset(name="blood_vessel_size", data=np.sort(blood_vessel_size))
    statistics.create_dataset(name="unknown_size", data=np.sort(unknown_size))

    data_summary.close()