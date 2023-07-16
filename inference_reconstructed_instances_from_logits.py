"""Inference of a model on a dataset. The model here is trained with reconstructed_model_progressive_supervised_unet.py"""

import gc
import time
import argparse

import tqdm

import torch.nn

import model_data_manager
import inference_reconstructed_base

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of a multiclass U-Net model")
    parser.add_argument("--reduction_logit_average", action="store_true",
                        help="Whether to average the logits. Default False.")
    parser.add_argument("--experts_only", action="store_true",
                        help="Whether to only use \"expert\" prediction. Default False.")
    model_data_manager.transform_add_argparse_arguments(parser, requires_model=False)
    args = parser.parse_args()
    input_data_loader, output_data_writer, model_path, subdata_entries, train_subdata_entries, val_subdata_entries = model_data_manager.transform_get_argparse_arguments(args, requires_model=False)
    assert input_data_loader.data_store is not None, "You must specify an input data which stores the logits, by the argument --original_data_name"

    computed = 0
    last_compute_print = 0
    ctime = time.time()

    image_radius = 384
    logits_group = input_data_loader.data_store["logits"]  # type: h5py.Group
    reduction_logit_average = args.reduction_logit_average
    experts_only = args.experts_only
    print("Computing now. Reduction logit average: {}    Experts only: {}".format(reduction_logit_average, experts_only))
    with tqdm.tqdm(total=len(subdata_entries)) as pbar:
        while computed < len(subdata_entries):
            tile_id = subdata_entries[computed]
            compute_end = computed + 1
            # Get logits from computed input data
            with torch.no_grad():
                img_helper = inference_reconstructed_base.Composite1024To512ImageInference()
                img_helper.load_logits_from_hdf(logits_group, tile_id)
                result = img_helper.obtain_predictions(reduction_logit_average, experts_only)

                masks = inference_reconstructed_base.get_instance_masks(result)
                image = inference_reconstructed_base.get_image_from_instances(masks)
                output_data_writer.write_image_data(subdata_entries[computed], image)

            gc.collect()
            pbar.update(compute_end - computed)
            computed = compute_end

            if computed - last_compute_print >= 100:
                print("Computed {} images in {:.2f} seconds".format(computed, time.time() - ctime))
                last_compute_print = computed
                ctime = time.time()

    input_data_loader.close()
    output_data_writer.close()
    print("Inference Complete")
