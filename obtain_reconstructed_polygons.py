import os
import json
import time
import collections

import numpy as np
import cv2

import model_data_manager
import config

import h5py

class Polygon:
    def __init__(self, polygon_type: str, bounds_x1: int, bounds_y1: int, bounding_box_mask: np.ndarray):
        self.polygon_type = polygon_type
        self.bounds_x1 = bounds_x1
        self.bounds_y1 = bounds_y1
        self.bounding_box_mask = bounding_box_mask
        self.wsi_tile_contains = {}

    def add_wsi(self, wsi_tile: str, wsi_mask: np.ndarray):
        self.wsi_tile_contains[wsi_tile] = wsi_mask

    def touches_top(self, wsi_tile: str):
        assert wsi_tile in self.wsi_tile_contains
        return self.wsi_tile_contains[wsi_tile][0, :].any()

    def touches_bottom(self, wsi_tile: str):
        assert wsi_tile in self.wsi_tile_contains
        return self.wsi_tile_contains[wsi_tile][-1, :].any()

    def touches_left(self, wsi_tile: str):
        assert wsi_tile in self.wsi_tile_contains
        return self.wsi_tile_contains[wsi_tile][:, 0].any()

    def touches_right(self, wsi_tile: str):
        assert wsi_tile in self.wsi_tile_contains
        return self.wsi_tile_contains[wsi_tile][:, -1].any()

    def touches(self, other: "Polygon", wsi_tile: str, other_wsi_tile: str, side:str="top"):
        """Returns True if the two polygons intersect, False otherwise."""
        assert self.polygon_type == other.polygon_type, "Polygon types do not match"
        assert wsi_tile in self.wsi_tile_contains
        assert other_wsi_tile in other.wsi_tile_contains

        combined_bounds_x1 = min(self.bounds_x1, other.bounds_x1)
        combined_bounds_y1 = min(self.bounds_y1, other.bounds_y1)
        combined_bounds_x2 = max(self.bounds_x1 + self.bounding_box_mask.shape[1],
                            other.bounds_x1 + other.bounding_box_mask.shape[1])
        combined_bounds_y2 = max(self.bounds_y1 + self.bounding_box_mask.shape[0],
                            other.bounds_y1 + other.bounding_box_mask.shape[0])

        combined_bounding_box_mask = np.zeros((combined_bounds_y2 - combined_bounds_y1, combined_bounds_x2 - combined_bounds_x1), dtype=bool)

        combined_bounding_box_mask[self.bounds_y1 - combined_bounds_y1:self.bounds_y1 - combined_bounds_y1 + self.bounding_box_mask.shape[0],
            self.bounds_x1 - combined_bounds_x1:self.bounds_x1 - combined_bounds_x1 + self.bounding_box_mask.shape[1]] = self.bounding_box_mask

        intersection = np.logical_and(combined_bounding_box_mask[
            other.bounds_y1 - combined_bounds_y1:other.bounds_y1 - combined_bounds_y1 + other.bounding_box_mask.shape[0],
            other.bounds_x1 - combined_bounds_x1:other.bounds_x1 - combined_bounds_x1 + other.bounding_box_mask.shape[1]],
                          other.bounding_box_mask)
        union = np.logical_or(combined_bounding_box_mask[
            other.bounds_y1 - combined_bounds_y1:other.bounds_y1 - combined_bounds_y1 + other.bounding_box_mask.shape[0],
            other.bounds_x1 - combined_bounds_x1:other.bounds_x1 - combined_bounds_x1 + other.bounding_box_mask.shape[1]],
                          other.bounding_box_mask)

        if np.count_nonzero(intersection) / np.count_nonzero(union) > 0.5:
            return True

        if side == "top":
            if not self.touches_top(wsi_tile) or not other.touches_bottom(other_wsi_tile):
                return False

            self_touching_pixels = self.wsi_tile_contains[wsi_tile][0, :]
            other_touching_pixels = other.wsi_tile_contains[other_wsi_tile][-1, :]

            intersection = np.logical_and(self_touching_pixels, other_touching_pixels)
            union = np.logical_or(self_touching_pixels, other_touching_pixels)

            if np.count_nonzero(intersection) / np.count_nonzero(union) <= 0.5:
                return False

            return True
        elif side == "bottom":
            if not self.touches_bottom(wsi_tile) or not other.touches_top(other_wsi_tile):
                return False

            self_touching_pixels = self.wsi_tile_contains[wsi_tile][-1, :]
            other_touching_pixels = other.wsi_tile_contains[other_wsi_tile][0, :]

            intersection = np.logical_and(self_touching_pixels, other_touching_pixels)
            union = np.logical_or(self_touching_pixels, other_touching_pixels)

            if np.count_nonzero(intersection) / np.count_nonzero(union) <= 0.5:
                return False

            return True
        elif side == "left":
            if not self.touches_left(wsi_tile) or not other.touches_right(other_wsi_tile):
                return False

            self_touching_pixels = self.wsi_tile_contains[wsi_tile][:, 0]
            other_touching_pixels = other.wsi_tile_contains[other_wsi_tile][:, -1]

            intersection = np.logical_and(self_touching_pixels, other_touching_pixels)
            union = np.logical_or(self_touching_pixels, other_touching_pixels)

            if np.count_nonzero(intersection) / np.count_nonzero(union) <= 0.5:
                return False

            return True
        elif side == "right":
            if not self.touches_right(wsi_tile) or not other.touches_left(other_wsi_tile):
                return False

            self_touching_pixels = self.wsi_tile_contains[wsi_tile][:, -1]
            other_touching_pixels = other.wsi_tile_contains[other_wsi_tile][:, 0]

            intersection = np.logical_and(self_touching_pixels, other_touching_pixels)
            union = np.logical_or(self_touching_pixels, other_touching_pixels)

            if np.count_nonzero(intersection) / np.count_nonzero(union) <= 0.5:
                return False

            return True
        else:
            raise ValueError("Invalid side argument")

    def merge_into(self, other: "Polygon"):
        """Merges the other polygon into self. The other polygon is not changed, while self is modified."""
        assert self.polygon_type == other.polygon_type, "Polygon types do not match"

        for wsi_tile in other.wsi_tile_contains:
            if wsi_tile in self.wsi_tile_contains:
                self.wsi_tile_contains[wsi_tile] = np.logical_or(self.wsi_tile_contains[wsi_tile], other.wsi_tile_contains[wsi_tile])
            else:
                self.wsi_tile_contains[wsi_tile] = other.wsi_tile_contains[wsi_tile]

        new_bounds_x1 = min(self.bounds_x1, other.bounds_x1)
        new_bounds_y1 = min(self.bounds_y1, other.bounds_y1)
        new_bounds_x2 = max(self.bounds_x1 + self.bounding_box_mask.shape[1], other.bounds_x1 + other.bounding_box_mask.shape[1])
        new_bounds_y2 = max(self.bounds_y1 + self.bounding_box_mask.shape[0], other.bounds_y1 + other.bounding_box_mask.shape[0])


        new_bounding_box_mask = np.zeros((new_bounds_y2 - new_bounds_y1, new_bounds_x2 - new_bounds_x1), dtype=bool)

        new_bounding_box_mask[self.bounds_y1 - new_bounds_y1:self.bounds_y1 - new_bounds_y1 + self.bounding_box_mask.shape[0],
            self.bounds_x1 - new_bounds_x1:self.bounds_x1 - new_bounds_x1 + self.bounding_box_mask.shape[1]] = self.bounding_box_mask

        new_bounding_box_mask[other.bounds_y1 - new_bounds_y1:other.bounds_y1 - new_bounds_y1 + other.bounding_box_mask.shape[0],
            other.bounds_x1 - new_bounds_x1:other.bounds_x1 - new_bounds_x1 + other.bounding_box_mask.shape[1]] = np.logical_or(
                new_bounding_box_mask[other.bounds_y1 - new_bounds_y1:other.bounds_y1 - new_bounds_y1 + other.bounding_box_mask.shape[0],
                    other.bounds_x1 - new_bounds_x1:other.bounds_x1 - new_bounds_x1 + other.bounding_box_mask.shape[1]], other.bounding_box_mask)

        self.bounds_x1 = new_bounds_x1
        self.bounds_y1 = new_bounds_y1
        self.bounding_box_mask = new_bounding_box_mask

    def save_to_hdf5(self, hdf5_group: h5py.Group, index: int):
        """Saves the polygon to the hdf5 file."""
        # remove holes
        num_labels, labels_im = cv2.connectedComponents(np.logical_not(self.bounding_box_mask).astype(np.uint8) * 255, connectivity=8)

        for k in range(1, num_labels):
            component_mask = (labels_im == k)
            exists_x = np.argwhere(np.any(component_mask, axis=0)).squeeze(-1)
            exists_y = np.argwhere(np.any(component_mask, axis=1)).squeeze(-1)

            bounds_x1, bounds_x2 = np.min(exists_x), np.max(exists_x)
            bounds_y1, bounds_y2 = np.min(exists_y), np.max(exists_y)

            if bounds_x1 > 0 and bounds_x2 < self.bounding_box_mask.shape[1] - 1 and bounds_y1 > 0 and bounds_y2 < self.bounding_box_mask.shape[0] - 1:
                self.bounding_box_mask[component_mask] = True

        # save data now
        polygon_group = hdf5_group.create_group(str(index))
        polygon_group.create_dataset("type", data=self.polygon_type, dtype=h5py.string_dtype())
        polygon_group.create_dataset("top_left_x", data=int(self.bounds_x1), dtype=np.int32)
        polygon_group.create_dataset("top_left_y", data=int(self.bounds_y1), dtype=np.int32)
        polygon_group.create_dataset("mask", data=self.bounding_box_mask, dtype=bool, compression="gzip", compression_opts=9, shape=self.bounding_box_mask.shape)

if not os.path.isfile(os.path.join(config.input_data_path, "polygons_refined.jsonl")):
    print("polygons_refined.jsonl does not exist in the data folder. Please run obtain_refined_polygons.py to remove the repeated polygons first.")
    quit(-1)

with open(os.path.join(config.input_data_path, "polygons_refined.jsonl")) as json_file:
    json_list = list(json_file)

all_polygon_masks = {}
for json_str in json_list:
    polygon_masks = json.loads(json_str)
    all_polygon_masks[polygon_masks["id"]] = polygon_masks["annotations"]

def obtain_reconstructed_polygons(wsi_id):
    polygons = []
    polygons_per_tile = collections.defaultdict(list)

    wsi_information = model_data_manager.data_information
    wsi_information = wsi_information.loc[wsi_information["source_wsi"] == wsi_id]

    for wsi_tile in wsi_information.index:
        x = wsi_information.loc[wsi_tile, "i"]
        y = wsi_information.loc[wsi_tile, "j"]

        if wsi_tile in all_polygon_masks:
            for polygon_mask in all_polygon_masks[wsi_tile]:
                polygon_type = polygon_mask["type"]
                polygon_coordinate_list = polygon_mask["coordinates"][0]  # This is a list of integer 2-tuples, representing the coordinates.

                polygon_tile_mask = np.zeros((512, 512, 1), dtype=np.uint8)
                cv2.fillPoly(polygon_tile_mask, [np.array(polygon_coordinate_list)], (1,))

                polygon_tile_mask = polygon_tile_mask.astype(bool).squeeze(-1)

                exists_x = np.argwhere(np.any(polygon_tile_mask, axis=0)).squeeze(-1)
                exists_y = np.argwhere(np.any(polygon_tile_mask, axis=1)).squeeze(-1)

                bounds_x1, bounds_x2 = np.min(exists_x), np.max(exists_x)
                bounds_y1, bounds_y2 = np.min(exists_y), np.max(exists_y)

                bounding_box_mask = polygon_tile_mask[bounds_y1:bounds_y2+1, bounds_x1:bounds_x2+1]

                top_left_x = x + bounds_x1
                top_left_y = y + bounds_y1

                polygon = Polygon(polygon_type, top_left_x, top_left_y, bounding_box_mask)
                polygon.add_wsi(wsi_tile, polygon_tile_mask)

                # check touching with existing polygons. first check if it touches boundary
                if polygon.touches_top(wsi_tile):
                    touching_tiles = wsi_information.loc[(y - 512 <= wsi_information["j"]) & (wsi_information["j"] <= y) &
                                                         (wsi_information["i"] == x)].index
                    for touching_tile in touching_tiles:
                        if (touching_tile != wsi_tile) and (touching_tile in polygons_per_tile):
                            idx = 0
                            while idx < len(polygons_per_tile[touching_tile]):
                                other_polygon = polygons_per_tile[touching_tile][idx]
                                if polygon.polygon_type == other_polygon.polygon_type and\
                                    other_polygon.touches_bottom(touching_tile) and\
                                    polygon.touches(other_polygon, wsi_tile, touching_tile, side="top"):

                                    polygon.merge_into(other_polygon)
                                    polygons.remove(other_polygon)
                                    for other_wsi_tile in other_polygon.wsi_tile_contains:
                                        polygons_per_tile[other_wsi_tile].remove(other_polygon)
                                else:
                                    idx += 1

                if polygon.touches_bottom(wsi_tile):
                    touching_tiles = wsi_information.loc[(y <= wsi_information["j"]) & (wsi_information["j"] <= y + 512) &
                                                         (wsi_information["i"] == x)].index
                    for touching_tile in touching_tiles:
                        if (touching_tile != wsi_tile) and (touching_tile in polygons_per_tile):
                            idx = 0
                            while idx < len(polygons_per_tile[touching_tile]):
                                other_polygon = polygons_per_tile[touching_tile][idx]
                                if polygon.polygon_type == other_polygon.polygon_type and\
                                    other_polygon.touches_top(touching_tile) and\
                                    polygon.touches(other_polygon, wsi_tile, touching_tile, side="bottom"):

                                    polygon.merge_into(other_polygon)
                                    polygons.remove(other_polygon)
                                    for other_wsi_tile in other_polygon.wsi_tile_contains:
                                        polygons_per_tile[other_wsi_tile].remove(other_polygon)
                                else:
                                    idx += 1

                if polygon.touches_left(wsi_tile):
                    touching_tiles = wsi_information.loc[(x - 512 <= wsi_information["i"]) & (wsi_information["i"] <= x) &
                                                         (wsi_information["j"] == y)].index
                    for touching_tile in touching_tiles:
                        if (touching_tile != wsi_tile) and (touching_tile in polygons_per_tile):
                            idx = 0
                            while idx < len(polygons_per_tile[touching_tile]):
                                other_polygon = polygons_per_tile[touching_tile][idx]
                                if polygon.polygon_type == other_polygon.polygon_type and\
                                    other_polygon.touches_right(touching_tile) and\
                                    polygon.touches(other_polygon, wsi_tile, touching_tile, side="left"):

                                    polygon.merge_into(other_polygon)
                                    polygons.remove(other_polygon)
                                    for other_wsi_tile in other_polygon.wsi_tile_contains:
                                        polygons_per_tile[other_wsi_tile].remove(other_polygon)
                                else:
                                    idx += 1

                if polygon.touches_right(wsi_tile):
                    touching_tiles = wsi_information.loc[(x <= wsi_information["i"]) & (wsi_information["i"] <= x + 512) &
                                                         (wsi_information["j"] == y)].index
                    for touching_tile in touching_tiles:
                        if (touching_tile != wsi_tile) and (touching_tile in polygons_per_tile):
                            idx = 0
                            while idx < len(polygons_per_tile[touching_tile]):
                                other_polygon = polygons_per_tile[touching_tile][idx]
                                if polygon.polygon_type == other_polygon.polygon_type and\
                                    other_polygon.touches_left(touching_tile) and\
                                    polygon.touches(other_polygon, wsi_tile, touching_tile, side="right"):

                                    polygon.merge_into(other_polygon)
                                    polygons.remove(other_polygon)
                                    for other_wsi_tile in other_polygon.wsi_tile_contains:
                                        polygons_per_tile[other_wsi_tile].remove(other_polygon)
                                else:
                                    idx += 1

                polygons.append(polygon)
                for tile in polygon.wsi_tile_contains:
                    polygons_per_tile[tile].append(polygon)

    return polygons

if not os.path.isdir("segmentation_reconstructed_data"):
    os.mkdir("segmentation_reconstructed_data")
if os.path.isfile(os.path.join("segmentation_reconstructed_data", "polygons_reconstructed.hdf5")):
    os.remove(os.path.join("segmentation_reconstructed_data", "polygons_reconstructed.hdf5"))

hdf_file = h5py.File(os.path.join("segmentation_reconstructed_data", "polygons_reconstructed.hdf5"), "w")

# reconstruct polygons now
for wsi_id in range(1, 15):
    if wsi_id != 5:
        group = hdf_file.create_group("wsi_{}".format(wsi_id))

        print("Starting to reconstruct polygons for wsi {}.".format(wsi_id))
        ctime = time.time()
        polygons = obtain_reconstructed_polygons(wsi_id)

        group.create_dataset("num_polygons", data=len(polygons), dtype=np.int32)
        for index in range(len(polygons)):
            polygon = polygons[index]
            polygon.save_to_hdf5(group, index)

        print("Finished reconstructing polygons for wsi {} in {} seconds.".format(wsi_id, time.time() - ctime))