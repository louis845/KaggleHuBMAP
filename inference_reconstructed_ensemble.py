"""Inference of a model on a dataset. The model here is trained with reconstructed_model_progressive_supervised_unet.py"""

import gc
import time
import argparse

import config

import numpy as np
import tqdm

import torch
import torch.nn

import model_data_manager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the ensemble probas by taking the average of a list of computed probas")
    parser.add_argument("--subdata", type=str, help="The subdata to transform the data on. If not specified, the whole data will be used.", required=True)
    parser.add_argument("--original_data_name", type=str, help="The names of the original data to use.", required=True, nargs="+")
    parser.add_argument("--transformed_data_name", type=str, required=True, help="The name of the transformed data.")
    args = parser.parse_args()

    if not model_data_manager.subdata_exists(args.subdata):
        print("Subdata {} does not exist".format(args.subdata))
        exit(1)
    subdata_entries = model_data_manager.get_subdata_entry_list(args.subdata)

    probas_data = []
    if len(args.original_data_name) == 0:
        print("No original data specified")
        exit(1)
    for original_data_name in args.original_data_name:
        if not model_data_manager.dataset_exists(original_data_name):
            print("Data probas {} does not exist".format(original_data_name))
            exit(1)
        probas_data.append(model_data_manager.get_dataset_dataloader(original_data_name))
        if "logits" in probas_data[-1].data_store:
            print("Data {} has logits, probably wrong data chosen?".format(original_data_name))
            exit(1)

    if not model_data_manager.request_data_create(args.transformed_data_name, "no_model"):
        print("Data {} already exists".format(args.transformed_data_name))
        exit(1)
    output_data_writer = model_data_manager.get_dataset_datawriter(args.transformed_data_name)

    computed = 0
    last_compute_print = 0
    ctime = time.time()


    with tqdm.tqdm(total=len(subdata_entries)) as pbar:
        while computed < len(subdata_entries):
            tile_id = subdata_entries[computed]
            compute_end = computed + 1
            # Get logits from computed input data
            with torch.no_grad():
                probas = []
                for input_data_loader in probas_data:
                    input_data = input_data_loader.get_image_data(tile_id)
                    assert input_data.dtype == np.float32, "Input data is not float32"
                    probas.append(torch.tensor(input_data, dtype=torch.float32, device=config.device))

                probas = torch.stack(probas, dim=0)
                probas = torch.mean(probas, dim=0)
                output_data_writer.write_image_data(tile_id, probas.cpu().numpy())

            gc.collect()
            torch.cuda.empty_cache()
            pbar.update(compute_end - computed)
            computed = compute_end

            if computed - last_compute_print >= 100:
                print("Computed {} images in {:.2f} seconds".format(computed, time.time() - ctime))
                last_compute_print = computed
                ctime = time.time()

    for input_data_loader in probas_data:
        input_data_loader.close()
    output_data_writer.close()
    print("Inference Complete")
