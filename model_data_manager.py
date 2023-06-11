import os
import json
import argparse
import shutil

import config

import torch
import pandas as pd
import numpy as np
import h5py
import cv2

model_dir = "models/"
transformed_data_dir = "transformed_data/"
subdata_dir = "subdata/"

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

if not os.path.exists(transformed_data_dir):
    os.mkdir(transformed_data_dir)

if not os.path.exists(subdata_dir):
    os.mkdir(subdata_dir)

data_information = pd.read_csv(os.path.join(config.input_data_path, "tile_meta.csv"), index_col=0)
entries_index_to_int_map = pd.Series(np.arange(len(data_information)), index=data_information.index)

def hdf5_load_in_memory(recursive):
    if isinstance(recursive, h5py.Dataset):
        return np.array(recursive)
    result = {}
    for key, val in recursive.items():
        result[key] = hdf5_load_in_memory(val)
    return result

class DatasetDataLoader():
    def __init__(self, dataset_name, load_in_memory=False):
        self.dataset_name = dataset_name
        if dataset_name is None:
            self.data_store = None
        else:
            self.data_store = h5py.File(os.path.join(transformed_data_dir, dataset_name, "data.hdf5"), "r")
        self.segmentation_store = h5py.File(os.path.join("segmentation_data", "data_summary.h5"), "r")

        if load_in_memory:
            self.data_store = hdf5_load_in_memory(self.data_store)
            self.segmentation_store = hdf5_load_in_memory(self.segmentation_store)

    def get_image_data(self, entry_name):
        if self.data_store is None:
            image_data = cv2.imread(os.path.join(config.input_data_path, "train", entry_name + ".tif"))
            # Convert to RGB
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            return image_data
        else:
            return np.array(self.data_store["image_data"][entry_name])

    def list_segmentation_masks(self):
        return self.segmentation_store["segmentation_data"][data_information.index[0]].keys()

    def get_segmentation_masks(self, entry_name):
        glomerulus = self.segmentation_store["segmentation_data"][entry_name]["glomerulus"]
        blood_vessel = self.segmentation_store["segmentation_data"][entry_name]["blood_vessel"]
        unknown = self.segmentation_store["segmentation_data"][entry_name]["unknown"]

        return {"glomerulus": np.array(glomerulus), "blood_vessel": np.array(blood_vessel), "unknown": np.array(unknown)}

    def get_segmentation_mask(self, entry_name, mask_name):
        return np.array(self.segmentation_store["segmentation_data"][entry_name][mask_name])

    def close(self):
        if self.data_store is not None:
            self.data_store.close()
        self.segmentation_store.close()

class DatasetDataWriter():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data_store = h5py.File(os.path.join(transformed_data_dir, dataset_name, "data.hdf5"), "w")

    def write_image_data(self, entry_name, image_data):
        if "image_data" not in self.data_store:
            self.data_store.create_group("image_data")

        if entry_name not in self.data_store["image_data"]:
            self.data_store["image_data"].create_dataset(entry_name, data=image_data, compression="gzip", compression_opts=9)
        else:
            self.data_store["image_data"][entry_name] = image_data

    def close(self):
        self.data_store.close()

def get_dataset_dataloader(dataset_name, load_in_memory: bool) -> DatasetDataLoader:
    if dataset_name is not None:
        assert dataset_exists(dataset_name)
    return DatasetDataLoader(dataset_name, load_in_memory)

def get_dataset_datawriter(dataset_name) -> DatasetDataWriter:
    return DatasetDataWriter(dataset_name)

def entry_list():
    return data_information.index

def get_entry_index_by_intid(intid: np.ndarray):
    return data_information.index[intid]

def get_intid_by_entry_index(entry_index: pd.Index):
    return entries_index_to_int_map.loc[entry_index].to_numpy()

def get_intersection(x: np.ndarray, y: np.ndarray):
    assert np.all(x[:-1] <= x[1:])
    assert np.all(y[:-1] <= y[1:])
    return x[np.searchsorted(y, x, side="left") < np.searchsorted(y, x, side="right")]

def model_exists(model_name):
    return os.path.exists(os.path.join(model_dir, model_name))

def save_model(model_name, model_saving_callback, json_metadata):
    model_saving_callback(os.path.join(model_dir, model_name))
    with open(os.path.join(model_dir, model_name, "metadata.json"), "w") as json_file:
        json.dump(json_metadata, json_file, indent=4)

def request_data_create(data_name, model_name, data_source=None):
    """
        Requests a new data creation routine. If data_source is None, then it is assumed that the data is in the given data.
    :param data_name: The new name of the data.
    :param model_name: The model name of the model that the data is created from.
    :param data_source: The original data this data is created from.
    :return: True / False, whether the data creation is successful.
    """
    if os.path.exists(os.path.join(transformed_data_dir, data_name)):
        return False
    os.mkdir(os.path.join(transformed_data_dir, data_name))
    if data_source is not None:
        total_history = os.path.join(transformed_data_dir, data_source, "data_history.json")
        with open(total_history) as json_file:
            data_history = json.load(json_file)["data_history"]
    else:
        data_history = []
    data_history.append({"model_name": model_name, "data_name": data_name})
    with open(os.path.join(transformed_data_dir, data_name, "data_history.json"), "w") as json_file:
        json.dump({"data_history": data_history}, json_file, indent=4)
    return True

def dataset_exists(data_name):
    return os.path.exists(os.path.join(transformed_data_dir, data_name))

def list_datasets():
    return os.listdir(transformed_data_dir)

def list_subdata():
    return [subdata[:-5] for subdata in os.listdir(subdata_dir)]

def get_subdata_entry_list(subdata):
    with open(os.path.join(subdata_dir, subdata + ".json")) as json_file:
        subdata_json = json.load(json_file)
    return subdata_json["entry_list"]

def model_add_argparse_arguments(parser, allow_missing_validation=False):
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model.")
    parser.add_argument("--dataset", type=str, help="The dataset to be trained on.")
    parser.add_argument("--train_subdata", type=str, required=True, help="The subdata to be used for training.")
    if allow_missing_validation:
        parser.add_argument("--val_subdata", type=str, help="The subdata to be used for validation.")
    else:
        parser.add_argument("--val_subdata", type=str, required=True, help="The subdata to be used for validation.")
    parser.add_argument("--prev_model_ckpt", type=str, help="The previous model to be loaded to continue training.")
    parser.add_argument("--load_in_memory", action="store_true", help="Whether to load the data in memory.")

def model_get_argparse_arguments(args, allow_missing_validation=False):
    model_name = args.model_name
    if model_exists(model_name):
        print("Model already exists! Pick another name.")
        quit()
    dataset = args.dataset
    if dataset is not None and not dataset_exists(dataset):
        print("Dataset does not exist! Pick another dataset. Available datasets:", os.listdir(transformed_data_dir))
        quit()
    elif dataset is None:
        print("No dataset is given. Training on the original data.")
    train_subdata = args.train_subdata
    if not subdata_exists(train_subdata):
        print("Training subdata does not exist! Pick another subdata. Available subdata:", os.listdir(subdata_dir))
        quit()
    val_subdata = args.val_subdata
    if allow_missing_validation:
        if val_subdata is not None and not subdata_exists(val_subdata):
            print("Validation subdata does not exist! Pick another subdata. Available subdata:", os.listdir(subdata_dir))
            quit()
    else:
        if not subdata_exists(val_subdata):
            print("Validation subdata does not exist! Pick another subdata. Available subdata:", os.listdir(subdata_dir))
            quit()
    prev_model_ckpt = args.prev_model_ckpt
    load_in_memory = args.load_in_memory

    # Load the json of train_subdata and val_subdata
    with open(os.path.join(subdata_dir, train_subdata + ".json")) as json_file:
        train_subdata_json = json.load(json_file)
    if val_subdata is not None:
        with open(os.path.join(subdata_dir, val_subdata + ".json")) as json_file:
            val_subdata_json = json.load(json_file)

    # Loop through the keys of the json dict and print it to the user
    print("Selected training and validation information:")
    for key in train_subdata_json.keys():
        if key == "entry_list":
            continue
        print("Train", key, ":  ", train_subdata_json[key])
        if val_subdata is not None:
            print("Valid", key, ":  ", val_subdata_json[key])

    print()
    print()

    training_entries = train_subdata_json["entry_list"]
    if val_subdata is not None:
        validation_entries = val_subdata_json["entry_list"]

    training_entries_num_id = get_intid_by_entry_index(pd.Index(training_entries))
    if val_subdata is not None:
        validation_entries_num_id = get_intid_by_entry_index(pd.Index(validation_entries))

    assert (training_entries_num_id[:-1] <= training_entries_num_id[1:]).all()
    if val_subdata is not None:
        assert (validation_entries_num_id[:-1] <= validation_entries_num_id[1:]).all()

        # Check if the intersection of the two sets is empty
        intersection_empty = np.sum(np.searchsorted(training_entries_num_id, validation_entries_num_id, side="left")
               < np.searchsorted(training_entries_num_id, validation_entries_num_id, side="right")) == 0

        if not intersection_empty:
            print("The training and validation sets are not disjoint! Pick another subdata.")
            quit()

    new_model_dir = os.path.join(model_dir, model_name)
    prev_model_dir = None
    if prev_model_ckpt is not None:
        prev_model_dir = os.path.join(model_dir, prev_model_ckpt)
    data_loader = get_dataset_dataloader(dataset, load_in_memory)

    if not os.path.exists(new_model_dir):
        os.mkdir(new_model_dir)

    if val_subdata is None:
        validation_entries = None

    extra_info = {"dataset": dataset, "train_subdata": train_subdata, "val_subdata": val_subdata, "previous_model": str(prev_model_ckpt)}

    return new_model_dir, data_loader, training_entries, validation_entries, prev_model_dir, extra_info

def transform_add_argparse_arguments(parser, requires_model=True):
    parser.add_argument("--original_data_name", type=str, help="The name of the original data. Don't type anything to use the original data.")
    parser.add_argument("--transformed_data_name", type=str, required=True, help="The name of the transformed data.")
    parser.add_argument("--subdata", type=str, help="The subdata to transform the data on. If not specified, the whole data will be used.")

    if requires_model:
        parser.add_argument("--model_name", type=str, required=True, help="The name of the model.")

def transform_get_argparse_arguments(args, requires_model=True) -> (DatasetDataLoader, DatasetDataWriter, str):
    input_data = args.original_data_name
    if input_data is None:
        print("Using default data.")
    else:
        if not dataset_exists(input_data):
            print("Dataset does not exist! Pick another dataset. Available datasets:", os.listdir(transformed_data_dir))
            quit()
    output_data = args.transformed_data_name

    if requires_model:
        model_name = args.model_name
        if not model_exists(model_name):
            print("Model does not exist! Pick another model. Available models:", os.listdir(model_dir))
            quit()

        success = request_data_create(output_data, model_name, input_data)
        if not success:
            print("Data already exists! Pick another name.")
            quit()
    else:
        success = request_data_create(output_data, "no_model", input_data)
        if not success:
            print("Data already exists! Pick another name.")
            quit()

    subdata = args.subdata
    subdata_entries = None
    if subdata is not None:
        if not subdata_exists(subdata):
            print("Subdata does not exist! Pick another subdata. Available subdata:", os.listdir(subdata_dir))
            quit()
        print("Using subdata {}".format(subdata))
        with open(os.path.join(subdata_dir, subdata + ".json")) as json_file:
            subdata_json = json.load(json_file)

        subdata_entries = subdata_json["entry_list"]

    input_data_loader = get_dataset_dataloader(input_data)
    output_data_writer = get_dataset_datawriter(output_data)

    if requires_model:
        model_path = os.path.join(model_dir, model_name)
        return input_data_loader, output_data_writer, model_path, subdata_entries
    else:
        return input_data_loader, output_data_writer, None, subdata_entries


def subdata_exists(subdata_name):
    return os.path.exists(os.path.join(subdata_dir, subdata_name + ".json"))

def generate_subdata_interactive(subdata_name):
    all_wsi = list(data_information["source_wsi"].unique())
    all_dataset = list(data_information["dataset"].unique())

    all_wsi.sort()
    all_dataset.sort()

    wsi_restriction = all_wsi.copy()
    dataset_restriction = all_dataset.copy()

    # Loop until the user types next. For each loop, ask the user to add or remove elements from the wsi_restriction. If the user types in a wsi that is not in the wsi_restriction,
    #  then add it. If the user types in a wsi that is in the wsi_restriction, then remove it. If the user types in a wsi that is not in the all_wsi, then print an error message.
    #  If the user types in next, then break out of the loop.
    while True:
        print("Current wsi restriction:", wsi_restriction)
        print("Type in a wsi to add or remove it from the restriction. Type in next to continue.")
        wsi = input("Enter wsi: ")
        if wsi == "next":
            break
        else:
            if not wsi.isdigit():
                print("Invalid wsi!")
                continue
            wsi = int(wsi)

            if wsi in all_wsi:
                if wsi in wsi_restriction:
                    wsi_restriction.remove(wsi)
                else:
                    wsi_restriction.append(wsi)
                    wsi_restriction.sort()
            else:
                print("Invalid wsi!")

    # Do the same for dataset
    while True:
        print("Current dataset restriction:", dataset_restriction)
        print("Type in a dataset to add or remove it from the restriction. Type in next to continue.")
        dataset = input("Enter dataset: ")
        if dataset == "next":
            break
        else:
            if not dataset.isdigit():
                print("Invalid dataset!")
                continue
            dataset = int(dataset)

            if dataset in all_dataset:
                if dataset in dataset_restriction:
                    dataset_restriction.remove(dataset)
                else:
                    dataset_restriction.append(dataset)
                    dataset_restriction.sort()
            else:
                print("Invalid dataset!")

    if len(wsi_restriction) == 0:
        print("The wsi restriction is empty! Returning to main routine...")
        return

    if len(dataset_restriction) == 0:
        print("The dataset restriction is empty! Returning to main routine...")
        return

    # Compute the intersection of the two restrictions (wsi, dataset)
    restriction_subset = data_information.loc[(data_information["source_wsi"].isin(wsi_restriction)) & (data_information["dataset"].isin(dataset_restriction))].index
    restriction_subset_int_ids = get_intid_by_entry_index(restriction_subset)

    # Choose stratification method
    stratification = None
    while True:
        user_input = input("Choose a stratification method (none, wsi, dataset):")
        if user_input == "none":
            stratification = None
            break
        elif user_input == "wsi":
            if len(wsi_restriction) == 1:
                print("The wsi restriction only has one wsi! Please choose another stratification method.")
                continue
            stratification = "wsi"
            break
        elif user_input == "dataset":
            if len(dataset_restriction) == 1:
                print("The dataset restriction only has one dataset! Please choose another stratification method.")
                continue
            stratification = "dataset"
            break
        else:
            print("Invalid stratification method! Choose from none, wsi, dataset.")

    # Choose prior subdata for exclusion
    prior_subdata = None
    while True:
        user_input = input("Choose a prior subdata to exclude. Type \"none\" for none, or type the name of the subdata for exclusion.")
        if user_input == "none":
            prior_subdata = None
            break
        elif subdata_exists(user_input):
            # Load the json file and check
            with open(os.path.join(subdata_dir, user_input + ".json")) as json_file:
                subdata_information = json.load(json_file)

            # If there is no stratification, check that the complement of the subdata is not empty. If there is stratification, check that the complement of the subdata is not empty for each stratum.
            if stratification is None:
                entry_list = pd.Index(subdata_information["entry_list"])
                entry_int_ids = get_intid_by_entry_index(entry_list)
                complement_mask = np.ones(len(data_information), dtype=bool)
                complement_mask[entry_int_ids] = False
                complement_int_ids = np.argwhere(complement_mask).flatten()

                complement_int_ids = get_intersection(complement_int_ids, restriction_subset_int_ids)

                if len(complement_int_ids) == 0:
                    print("The complement of the subdata is empty! Please choose another prior subdata.")
                    continue
            else:
                entry_list = pd.Index(subdata_information["entry_list"])
                entry_int_ids = get_intid_by_entry_index(entry_list)
                complement_mask = np.ones(len(data_information), dtype=bool)
                complement_mask[entry_int_ids] = False
                complement_subdata = data_information.loc[complement_mask]

                if stratification == "wsi":
                    success = True
                    for wsi in wsi_restriction:
                        if (data_information["source_wsi"].loc[restriction_subset] == wsi).sum() > 0:
                            complement_wsi_group = complement_subdata.loc[complement_subdata["source_wsi"] == wsi].index
                            complement_wsi_group_int_ids = get_intid_by_entry_index(complement_wsi_group)
                            complement_wsi_group_intersection = get_intersection(complement_wsi_group_int_ids, restriction_subset_int_ids)

                            if len(complement_wsi_group_intersection) < 5:
                                print("The complement of the subdata is too small for wsi", wsi, "! Please choose another prior subdata.")
                                success = False
                                break
                    if not success:
                        continue
                elif stratification == "dataset":
                    success = True
                    for dataset in dataset_restriction:
                        if (data_information["dataset"].loc[restriction_subset] == dataset).sum() > 0:
                            complement_dataset_group = complement_subdata.loc[complement_subdata["dataset"] == dataset].index
                            complement_dataset_group_int_ids = get_intid_by_entry_index(complement_dataset_group)
                            complement_dataset_group_intersection = get_intersection(complement_dataset_group_int_ids, restriction_subset_int_ids)

                            if len(complement_dataset_group_intersection) < 50:
                                print("The complement of the subdata is too small for dataset", dataset, "! Please choose another prior subdata.")
                                success = False
                                break
                    if not success:
                        continue
            prior_subdata = user_input
            break
        else:
            print("Subdata does not exist! Please choose another subdata (or type none).")

    # Choose number of subdata, as a ratio of len(data_information), between 0.0 and 1.0. Loop until the user enters a valid number in the range.
    while True:
        user_input = input("Choose a number of subdata, as a ratio of len(data_information), between 0.0 and 1.0:")
        try:
            number_of_subdata = float(user_input)
            if number_of_subdata < 0.0 or number_of_subdata > 1.0:
                print("Number of subdata must be between 0.0 and 1.0!")
                continue
            break
        except ValueError:
            print("Invalid number!")

    # Sample now.
    sampled_num_ids = None
    rng = np.random.default_rng()
    if prior_subdata is None:
        if stratification is None:
            sampled_num_ids = np.unique(rng.choice(restriction_subset_int_ids, int(number_of_subdata * len(restriction_subset_int_ids)), replace=False))
        else:
            if stratification == "wsi":
                sampled_num_ids = []
                for wsi in wsi_restriction:
                    wsi_mask = data_information["source_wsi"] == wsi
                    wsi_int_ids = np.argwhere(wsi_mask).flatten()
                    wsi_int_ids = get_intersection(wsi_int_ids, restriction_subset_int_ids)
                    sampled_num_ids.append(np.unique(rng.choice(wsi_int_ids, int(number_of_subdata * len(wsi_int_ids)), replace=False)))

                sampled_num_ids = np.unique(np.concatenate(sampled_num_ids))
            elif stratification == "dataset":
                sampled_num_ids = []
                for dataset in dataset_restriction:
                    dataset_mask = data_information["dataset"] == dataset
                    dataset_int_ids = np.argwhere(dataset_mask).flatten()
                    dataset_int_ids = get_intersection(dataset_int_ids, restriction_subset_int_ids)
                    sampled_num_ids.append(np.unique(rng.choice(dataset_int_ids, int(number_of_subdata * len(dataset_int_ids)), replace=False)))

                sampled_num_ids = np.unique(np.concatenate(sampled_num_ids))
    else:
        with open(os.path.join(subdata_dir, prior_subdata + ".json")) as json_file:
            subdata_information = json.load(json_file)

        entry_list = pd.Index(subdata_information["entry_list"])
        entry_int_ids = get_intid_by_entry_index(entry_list)
        complement_mask = np.ones(len(data_information), dtype=bool)
        complement_mask[entry_int_ids] = False
        complement_int_ids = np.argwhere(complement_mask).flatten()
        complement_int_ids = get_intersection(complement_int_ids, restriction_subset_int_ids)

        if stratification is None:
            if number_of_subdata == 1.0:
                sampled_num_ids = complement_int_ids
            else:
                sampled_num_ids = np.unique(rng.choice(complement_int_ids, int(number_of_subdata * len(data_information)), replace=False))
        else:
            if stratification == "wsi":
                sampled_num_ids = []
                for wsi in wsi_restriction:
                    wsi_mask = data_information["source_wsi"] == wsi
                    wsi_int_ids = np.argwhere(wsi_mask).flatten()
                    intersection = get_intersection(wsi_int_ids, complement_int_ids)
                    if number_of_subdata == 1.0:
                        sampled_num_ids.append(intersection)
                    else:
                        sampled_num_ids.append(np.unique(rng.choice(intersection, int(number_of_subdata * len(wsi_int_ids)), replace=False)))

                sampled_num_ids = np.unique(np.concatenate(sampled_num_ids))
            elif stratification == "dataset":
                sampled_num_ids = []
                for dataset in dataset_restriction:
                    dataset_mask = data_information["dataset"] == dataset
                    dataset_int_ids = np.argwhere(dataset_mask).flatten()
                    intersection = get_intersection(dataset_int_ids, complement_int_ids)
                    if number_of_subdata == 1.0:
                        sampled_num_ids.append(intersection)
                    else:
                        sampled_num_ids.append(np.unique(rng.choice(intersection, int(number_of_subdata * len(dataset_int_ids)), replace=False)))

                sampled_num_ids = np.unique(np.concatenate(sampled_num_ids))

    # Save the subdata.
    entry_list = list(get_entry_index_by_intid(sampled_num_ids))

    wsi_restriction = [int(wsi) for wsi in wsi_restriction]
    dataset_restriction = [int(dataset) for dataset in dataset_restriction]

    subdata_information = {
        "entry_list": entry_list,
        "stratification": stratification if stratification is not None else "none",
        "wsi_restriction": list(wsi_restriction),
        "dataset_restriction": list(dataset_restriction),
        "prior_subdata": prior_subdata if prior_subdata is not None else "none",
        "number_of_subdata": float(number_of_subdata),
    }

    with open(os.path.join(subdata_dir, subdata_name + ".json"), "w") as json_file:
        json.dump(subdata_information, json_file, indent=4)

if __name__ == "__main__":
    """Interactive command line interface to manage models / datasets.
    Commands: help, list model, list dataset, list subdata, remove model, remove dataset, remove subdata, generate subdata, quit"""
    parser = argparse.ArgumentParser(description="Interactive command line interface to manage models / datasets.")

    args = parser.parse_args()

    # Use the Python print() and input() functions to interact with the user.

    while True:
        command = input("Enter command: ")
        if command == "help":
            print("Commands: help, list model, list dataset, list subdata, remove model, remove dataset, remove subdata, generate subdata, quit")
        elif command == "list model":
            print("Models:")
            for model_name in os.listdir(model_dir):
                print("    ", model_name)
        elif command == "list dataset":
            print("Datasets:")
            for dataset_name in os.listdir(transformed_data_dir):
                print("    ", dataset_name)
        elif command == "list subdata":
            print("Subdata:")
            for dataset_name in os.listdir(transformed_data_dir):
                print("    ", dataset_name)
                for subdata_name in os.listdir(os.path.join(transformed_data_dir, dataset_name)):
                    print("        ", subdata_name)
        elif command == "remove model":
            model_name = input("Enter model name: ")
            if model_exists(model_name):
                shutil.rmtree(os.path.join(model_dir, model_name))
            else:
                print("Model does not exist!")
        elif command == "remove dataset":
            dataset_name = input("Enter dataset name: ")
            if os.path.exists(os.path.join(transformed_data_dir, dataset_name)):
                shutil.rmtree(os.path.join(transformed_data_dir, dataset_name))
            else:
                print("Dataset does not exist!")
        elif command == "remove subdata":
            subdata_name = input("Enter subdata name: ")
            if os.path.exists(os.path.join(subdata_dir, subdata_name + ".json")):
                os.remove(os.path.join(subdata_dir, subdata_name + ".json"))
            else:
                print("Subdata does not exist!")
        elif command == "generate subdata":
            subdata_name = input("Enter subdata name: ")
            if subdata_exists(subdata_name):
                print("Subdata already exists!")
            else:
                generate_subdata_interactive(subdata_name)
        elif command == "quit":
            break
        else:
            print("Invalid command! Type 'help' for a list of commands.")