import os
import json
import time

import numpy as np
import cv2

import model_data_manager
import config

import shapely.geometry

if not os.path.isfile(os.path.join(config.input_data_path, "polygons_refined.jsonl")):
    print("polygons_refined.jsonl does not exist in the data folder. Please run obtain_refined_polygons.py to remove the repeated polygons first.")
    quit(-1)

with open(os.path.join(config.input_data_path, "polygons_refined.jsonl")) as json_file:
    json_list = list(json_file)

all_polygon_masks = {}
for json_str in json_list:
    polygon_masks = json.loads(json_str)
    all_polygon_masks[polygon_masks["id"]] = polygon_masks["annotations"]


def shift_polygon(polygon_coordinates, x, y):
    for k in range(len(polygon_coordinates)):
        polygon_coordinates[k][0] += x
        polygon_coordinates[k][1] += y

def obtain_bounding_box(polygon_coordinates):
    x_min = polygon_coordinates[0][0]
    x_max = polygon_coordinates[0][0]
    y_min = polygon_coordinates[0][1]
    y_max = polygon_coordinates[0][1]
    for k in range(1, len(polygon_coordinates)):
        x_min = min(x_min, polygon_coordinates[k][0])
        x_max = max(x_max, polygon_coordinates[k][0])
        y_min = min(y_min, polygon_coordinates[k][1])
        y_max = max(y_max, polygon_coordinates[k][1])
    return x_min, x_max, y_min, y_max

def polygons_intersect(polygon_coordinates1, polygon_coordinates2):
    x_min1, x_max1, y_min1, y_max1 = obtain_bounding_box(polygon_coordinates1)
    x_min2, x_max2, y_min2, y_max2 = obtain_bounding_box(polygon_coordinates2)
    if x_min1 > x_max2 or x_max1 < x_min2 or y_min1 > y_max2 or y_max1 < y_min2:
        return False

    # Use shapely to check if the polygons intersect
    polygon1 = shapely.geometry.Polygon(polygon_coordinates1)
    polygon2 = shapely.geometry.Polygon(polygon_coordinates2)

    return polygon1.intersects(polygon2)

def obtain_reconstructed_polygons(wsi_id):
    polygons = []

    wsi_information = model_data_manager.data_information
    wsi_information = wsi_information.loc[wsi_information["source_wsi"] == wsi_id]
    width = int(wsi_information["i"].max() + 512)
    height = int(wsi_information["j"].max() + 512)

    for wsi_tile in wsi_information.index:
        x = wsi_information.loc[wsi_tile, "i"]
        y = wsi_information.loc[wsi_tile, "j"]

        if wsi_tile in all_polygon_masks:
            for polygon_mask in all_polygon_masks[wsi_tile]:
                polygon_type = polygon_mask["type"]
                polygon_coordinate_list = polygon_mask["coordinates"][0]  # This is a list of integer 2-tuples, representing the coordinates.
                shift_polygon(polygon_coordinate_list, x, y)

                # check if this polygon intersects with any of the existing polygons
                intersecting_polygons = []
                for idx in range(len(polygons)):
                    polygon = polygons[idx]["coordinates"]
                    if polygons_intersect(polygon, polygon_coordinate_list):
                        intersecting_polygons.append({"polygon":polygon, "idx":idx})

                # if there is no intersecting polygon, add this polygon to the list
                if len(intersecting_polygons) == 0:
                    polygons.append({"type": polygon_type, "coordinates":polygon_coordinate_list})

                    # if there are intersecting polygons, merge them into one polygon
                elif len(intersecting_polygons) > 1:
                    print(wsi_tile)

                    new_polygon_shapely = shapely.geometry.Polygon(polygon_coordinate_list)
                    for polygon_info in intersecting_polygons:
                        polygon = polygon_info["polygon"]
                        new_polygon_shapely = new_polygon_shapely.union(shapely.geometry.Polygon(polygon))

                        if type(new_polygon_shapely) == shapely.geometry.Polygon:
                            # Loop through all the holes in the polygon
                            for hole in new_polygon_shapely.interiors:
                                hole_area = shapely.geometry.Polygon(hole).area  # Get the area of the hole
                                print(f"Hole detected (will be removed)! Area: {hole_area:.2f}")

                            # Remove all holes in the polygon
                            new_polygon_shapely = shapely.geometry.Polygon(new_polygon_shapely.exterior)
                        elif type(new_polygon_shapely) == shapely.geometry.GeometryCollection:
                            assert len(new_polygon_shapely.geoms) == 2, "The union of two polygons should be either a Polygon or a GeometryCollection with two elements."
                            for polygon in new_polygon_shapely.geoms:
                                if type(polygon) == shapely.geometry.Polygon:
                                    new_polygon_shapely = shapely.geometry.Polygon(polygon.exterior)

                        polygons.pop(polygon_info["idx"])

                        print("Hole removed:", new_polygon_shapely)

                        assert not type(new_polygon_shapely) is shapely.geometry.MultiPolygon, "The union of two polygons should not be a MultiPolygon."
                        assert not type(new_polygon_shapely) is shapely.geometry.GeometryCollection, "The union of two polygons should not be a GeometryCollection."

                    new_polygon = list(new_polygon_shapely.exterior.coords)

                    polygons.append({"type": polygon_type, "coordinates":new_polygon})

    return polygons

ctime = time.time()

# reconstruct polygons now
reconstructed_polygons = {}  # key: wsi_id, value: list of polygons
for wsi_id in range(1, 15):
    if wsi_id != 5:
        print("Starting to reconstruct polygons for wsi {}.".format(wsi_id))
        reconstructed_polygons[wsi_id] = obtain_reconstructed_polygons(wsi_id)
        print("Finished reconstructing polygons for wsi {} in {} seconds.".format(wsi_id, time.time() - ctime))
        ctime = time.time()

# save the reconstructed polygons
with open(os.path.join(config.input_data_path, "polygons_reconstructed.jsonl"), "w") as json_file:
    json.dump(reconstructed_polygons, json_file, separators=(",", ":"))