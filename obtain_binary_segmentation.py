import os
import shutil
import time
import json

import numpy as np
import pandas as pd
import cv2
import h5py

import config

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
very_large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
super_large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
def get_mask_information(mask):
    mask = np.squeeze(mask, axis=-1).copy()
    large_mask = cv2.dilate(mask, large_kernel, iterations=1)
    very_large_mask = cv2.dilate(mask, very_large_kernel, iterations=1)
    super_large_mask = cv2.dilate(mask, super_large_kernel, iterations=1)

    erosion_history = []
    # Erode with a 3x3 circular kernel
    while True:
        erosion_history.append(mask)
        mask = cv2.erode(mask, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)

        if np.sum(mask) == 0:
            break

        mask_255 = mask * 255

        num_labels, labels_im = cv2.connectedComponents(mask_255, connectivity=8)

        if num_labels > 2:
            break

    erode5_mask = erosion_history[min(5, len(erosion_history) - 1)]
    erode10_mask = erosion_history[min(10, len(erosion_history) - 1)]
    erode25_percent_mask = erosion_history[int(0.25 * (len(erosion_history) - 1))]
    erode50_percent_mask = erosion_history[int(0.50 * (len(erosion_history) - 1))]
    erode75_percent_mask = erosion_history[int(0.75 * (len(erosion_history) - 1))]

    assert np.sum(erode5_mask * (1 - erosion_history[0])) == 0
    assert np.sum(erode10_mask * (1 - erosion_history[0])) == 0
    assert np.sum(erode25_percent_mask * (1 - erosion_history[0])) == 0
    assert np.sum(erode50_percent_mask * (1 - erosion_history[0])) == 0
    assert np.sum(erode75_percent_mask * (1 - erosion_history[0])) == 0

    assert large_mask.shape == (512, 512)
    assert very_large_mask.shape == (512, 512)
    assert super_large_mask.shape == (512, 512)
    assert erode5_mask.shape == (512, 512)
    assert erode10_mask.shape == (512, 512)
    assert erode25_percent_mask.shape == (512, 512)
    assert erode50_percent_mask.shape == (512, 512)
    assert erode75_percent_mask.shape == (512, 512)

    return {
        "large_mask": np.expand_dims(large_mask, axis=-1),
        "very_large_mask": np.expand_dims(very_large_mask, axis=-1),
        "super_large_mask": np.expand_dims(super_large_mask, axis=-1),
        "erode5_mask": np.expand_dims(erode5_mask, axis=-1),
        "erode10_mask": np.expand_dims(erode10_mask, axis=-1),
        "erode25_percent_mask": np.expand_dims(erode25_percent_mask, axis=-1),
        "erode50_percent_mask": np.expand_dims(erode50_percent_mask, axis=-1),
        "erode75_percent_mask": np.expand_dims(erode75_percent_mask, axis=-1)
    }


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

            glomerulus_mask_large = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            blood_vessel_mask_large = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            unknown_mask_large = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)

            glomerulus_mask_very_large = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            blood_vessel_mask_very_large = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            unknown_mask_very_large = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)

            glomerulus_mask_super_large = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            blood_vessel_mask_super_large = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            unknown_mask_super_large = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)

            glomerulus_mask_erode5 = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            blood_vessel_mask_erode5 = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            unknown_mask_erode5 = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)

            glomerulus_mask_erode10 = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            blood_vessel_mask_erode10 = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            unknown_mask_erode10 = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)

            glomerulus_mask_erode25_percent = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            blood_vessel_mask_erode25_percent = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            unknown_mask_erode25_percent = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)

            glomerulus_mask_erode50_percent = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            blood_vessel_mask_erode50_percent = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            unknown_mask_erode50_percent = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)

            glomerulus_mask_erode75_percent = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            blood_vessel_mask_erode75_percent = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
            unknown_mask_erode75_percent = np.zeros(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)

            for polygon_mask in all_polygon_masks[entry]:
                # The color depends on the type, default unknown color = blue
                polygon_coordinate_list = polygon_mask["coordinates"][0]  # This is a list of integer 2-tuples, representing the coordinates.
                mask = np.zeros_like(glomerulus_mask, dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(polygon_coordinate_list)], (1,))

                mask_deviants = get_mask_information(mask)
                if polygon_mask["type"] == "glomerulus":
                    glomerulus_mask = np.bitwise_or(glomerulus_mask, mask)
                    glomerulus_mask_large = np.bitwise_or(glomerulus_mask_large, mask_deviants["large_mask"])
                    glomerulus_mask_very_large = np.bitwise_or(glomerulus_mask_very_large, mask_deviants["very_large_mask"])
                    glomerulus_mask_super_large = np.bitwise_or(glomerulus_mask_super_large, mask_deviants["super_large_mask"])
                    glomerulus_mask_erode5 = np.bitwise_or(glomerulus_mask_erode5, mask_deviants["erode5_mask"])
                    glomerulus_mask_erode10 = np.bitwise_or(glomerulus_mask_erode10, mask_deviants["erode10_mask"])
                    glomerulus_mask_erode25_percent = np.bitwise_or(glomerulus_mask_erode25_percent, mask_deviants["erode25_percent_mask"])
                    glomerulus_mask_erode50_percent = np.bitwise_or(glomerulus_mask_erode50_percent, mask_deviants["erode50_percent_mask"])
                    glomerulus_mask_erode75_percent = np.bitwise_or(glomerulus_mask_erode75_percent, mask_deviants["erode75_percent_mask"])

                    num_glomerulus += 1
                    glomerulus_size.append(np.sum(mask))
                elif polygon_mask["type"] == "blood_vessel":
                    blood_vessel_mask = np.bitwise_or(blood_vessel_mask, mask)
                    blood_vessel_mask_large = np.bitwise_or(blood_vessel_mask_large, mask_deviants["large_mask"])
                    blood_vessel_mask_very_large = np.bitwise_or(blood_vessel_mask_very_large, mask_deviants["very_large_mask"])
                    blood_vessel_mask_super_large = np.bitwise_or(blood_vessel_mask_super_large, mask_deviants["super_large_mask"])
                    blood_vessel_mask_erode5 = np.bitwise_or(blood_vessel_mask_erode5, mask_deviants["erode5_mask"])
                    blood_vessel_mask_erode10 = np.bitwise_or(blood_vessel_mask_erode10, mask_deviants["erode10_mask"])
                    blood_vessel_mask_erode25_percent = np.bitwise_or(blood_vessel_mask_erode25_percent, mask_deviants["erode25_percent_mask"])
                    blood_vessel_mask_erode50_percent = np.bitwise_or(blood_vessel_mask_erode50_percent, mask_deviants["erode50_percent_mask"])
                    blood_vessel_mask_erode75_percent = np.bitwise_or(blood_vessel_mask_erode75_percent, mask_deviants["erode75_percent_mask"])

                    num_blood_vessel += 1
                    blood_vessel_size.append(np.sum(mask))
                else:
                    unknown_mask = np.bitwise_or(unknown_mask, mask)
                    unknown_mask_large = np.bitwise_or(unknown_mask_large, mask_deviants["large_mask"])
                    unknown_mask_very_large = np.bitwise_or(unknown_mask_very_large, mask_deviants["very_large_mask"])
                    unknown_mask_super_large = np.bitwise_or(unknown_mask_super_large, mask_deviants["super_large_mask"])
                    unknown_mask_erode5 = np.bitwise_or(unknown_mask_erode5, mask_deviants["erode5_mask"])
                    unknown_mask_erode10 = np.bitwise_or(unknown_mask_erode10, mask_deviants["erode10_mask"])
                    unknown_mask_erode25_percent = np.bitwise_or(unknown_mask_erode25_percent, mask_deviants["erode25_percent_mask"])
                    unknown_mask_erode50_percent = np.bitwise_or(unknown_mask_erode50_percent, mask_deviants["erode50_percent_mask"])
                    unknown_mask_erode75_percent = np.bitwise_or(unknown_mask_erode75_percent, mask_deviants["erode75_percent_mask"])

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

            entry_h5.create_dataset(name="glomerulus", data = (np.squeeze(glomerulus_mask, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="blood_vessel", data = (np.squeeze(blood_vessel_mask, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="unknown", data = (np.squeeze(unknown_mask, axis=-1) > 0), compression="gzip", compression_opts=9)

            entry_h5.create_dataset(name="glomerulus_large", data = (np.squeeze(glomerulus_mask_large, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="blood_vessel_large", data = (np.squeeze(blood_vessel_mask_large, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="unknown_large", data = (np.squeeze(unknown_mask_large, axis=-1) > 0), compression="gzip", compression_opts=9)

            entry_h5.create_dataset(name="glomerulus_very_large", data = (np.squeeze(glomerulus_mask_very_large, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="blood_vessel_very_large", data = (np.squeeze(blood_vessel_mask_very_large, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="unknown_very_large", data = (np.squeeze(unknown_mask_very_large, axis=-1) > 0), compression="gzip", compression_opts=9)

            entry_h5.create_dataset(name="glomerulus_super_large", data = (np.squeeze(glomerulus_mask_super_large, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="blood_vessel_super_large", data = (np.squeeze(blood_vessel_mask_super_large, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="unknown_super_large", data = (np.squeeze(unknown_mask_super_large, axis=-1) > 0), compression="gzip", compression_opts=9)

            entry_h5.create_dataset(name="glomerulus_erode5", data = (np.squeeze(glomerulus_mask_erode5, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="blood_vessel_erode5", data = (np.squeeze(blood_vessel_mask_erode5, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="unknown_erode5", data = (np.squeeze(unknown_mask_erode5, axis=-1) > 0), compression="gzip", compression_opts=9)

            entry_h5.create_dataset(name="glomerulus_erode10", data = (np.squeeze(glomerulus_mask_erode10, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="blood_vessel_erode10", data = (np.squeeze(blood_vessel_mask_erode10, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="unknown_erode10", data = (np.squeeze(unknown_mask_erode10, axis=-1) > 0), compression="gzip", compression_opts=9)

            entry_h5.create_dataset(name="glomerulus_erode25_percent", data = (np.squeeze(glomerulus_mask_erode25_percent, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="blood_vessel_erode25_percent", data = (np.squeeze(blood_vessel_mask_erode25_percent, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="unknown_erode25_percent", data = (np.squeeze(unknown_mask_erode25_percent, axis=-1) > 0), compression="gzip", compression_opts=9)

            entry_h5.create_dataset(name="glomerulus_erode50_percent", data = (np.squeeze(glomerulus_mask_erode50_percent, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="blood_vessel_erode50_percent", data = (np.squeeze(blood_vessel_mask_erode50_percent, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="unknown_erode50_percent", data = (np.squeeze(unknown_mask_erode50_percent, axis=-1) > 0), compression="gzip", compression_opts=9)

            entry_h5.create_dataset(name="glomerulus_erode75_percent", data = (np.squeeze(glomerulus_mask_erode75_percent, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="blood_vessel_erode75_percent", data = (np.squeeze(blood_vessel_mask_erode75_percent, axis=-1) > 0), compression="gzip", compression_opts=9)
            entry_h5.create_dataset(name="unknown_erode75_percent", data = (np.squeeze(unknown_mask_erode75_percent, axis=-1) > 0), compression="gzip", compression_opts=9)

            # Save the masks as 3 png files using cv2.imwrite
            cv2.imwrite(os.path.join("segmentation_data", entry, "glomerulus.png"), glomerulus_mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join("segmentation_data", entry, "blood_vessel.png"), blood_vessel_mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join("segmentation_data", entry, "unknown.png"), unknown_mask.astype(np.uint8) * 255)

        count += 1
        if count % 100 == 0:
            print("----------   Processed {} images. Time elapsed: {}   ----------".format(count, time.time() - ctime))
            ctime = time.time()

    # Save the statistics. Use np sort to sort glomerulus_size, blood_vessel_size, unknown_size, and save them into a statistics.npz file
    np.savez_compressed(os.path.join("segmentation_data", "statistics.npz"),
                        glomerulus_size=np.sort(glomerulus_size),
                        blood_vessel_size=np.sort(blood_vessel_size),
                        unknown_size=np.sort(unknown_size))

    statistics = data_summary.create_group("statistics")
    statistics.create_dataset(name="glomerulus_size", data=np.sort(glomerulus_size), compression="gzip", compression_opts=9)
    statistics.create_dataset(name="blood_vessel_size", data=np.sort(blood_vessel_size), compression="gzip", compression_opts=9)
    statistics.create_dataset(name="unknown_size", data=np.sort(unknown_size), compression="gzip", compression_opts=9)

    data_summary.close()