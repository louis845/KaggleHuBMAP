"""Inference of a model on a dataset. The model here is trained with model_simple_unet.py"""

import gc
import os
import time
import argparse
import json

import config

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn
import torchvision
import torchvision.transforms.functional

import model_data_manager
import model_unet_progressive

def reset_running_stats(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of a simple U-Net model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use. Default 2.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true", help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels to use. Default 3.")
    parser.add_argument("--preinit_batch_stats", action="store_true", help="Whether to preinitialize batch statistics. Default False.")

    image_width = 512
    image_height = 512

    model_data_manager.transform_add_argparse_arguments(parser)

    args = parser.parse_args()

    input_data_loader, output_data_writer, model_path, subdata_entries, train_subdata_entries, val_subdata_entries = model_data_manager.transform_get_argparse_arguments(args)

    specific_inference = subdata_entries is not None

    if subdata_entries is None:
        subdata_entries = list(model_data_manager.data_information)

    model = model_unet_progressive.UNetClassifier(hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                           use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                           in_channels=args.in_channels).to(device=config.device)

    model_checkpoint_path = os.path.join(model_path, "model.pt")

    model.load_state_dict(torch.load(model_checkpoint_path))

    batch_size = args.batch_size
    preinit_batch_stats = args.preinit_batch_stats

    computed = 0
    last_compute_print = 0
    ctime = time.time()

    if specific_inference:
        true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0
        true_negative_crude, true_positive_crude, false_negative_crude, false_positive_crude = 0, 0, 0, 0

    if preinit_batch_stats:
        image_pixels_round = 2 ** args.pyramid_height
        model.eval()

    while computed < len(subdata_entries):
        # Test the model
        with torch.no_grad():
            compute_end = min(computed + batch_size, len(subdata_entries))
            inference_batch = torch.zeros((compute_end - computed, 3, image_height, image_width), dtype=torch.float32, device=config.device)

            for k in range(computed, compute_end):
                inference_batch[k - computed, :, :, :] = torch.tensor(input_data_loader.get_image_data(subdata_entries[k]), dtype=torch.float32, device=config.device).permute(2, 0, 1)

            """if preinit_batch_stats:
                reset_running_stats(model)
                for k in range(24):
                    # flip the images
                    if np.random.uniform(0, 1) < 0.5:
                        inference_batch2 = torch.flip(inference_batch, dims=[3])
                    else:
                        inference_batch2 = inference_batch

                    # Rotate the images
                    angle_in_deg = np.random.uniform(0, 360)
                    inference_batch2 = torchvision.transforms.functional.rotate(inference_batch2, angle_in_deg)

                    rads = np.radians(angle_in_deg % 90.0)
                    lims = 0.5 / (np.sin(rads) + np.cos(rads))
                    # Restrict to (centerx - imagewidth * lims, centery - imageheight * lims) to (centerx + imagewidth * lims, centery + imageheight * lims)
                    ymin = int(image_height // 2 - image_height * lims)
                    ymax = int(image_height // 2 + image_height * lims)
                    xmin = int(image_width // 2 - image_width * lims)
                    xmax = int(image_width // 2 + image_width * lims)

                    xmax = image_pixels_round * ((xmax - xmin) // image_pixels_round) + xmin
                    ymax = image_pixels_round * ((ymax - ymin) // image_pixels_round) + ymin

                    inference_batch2 = inference_batch2[:, :, ymin:ymax, xmin:xmax]

                    model(inference_batch2)"""


            pred = model(inference_batch)
            pred_mask = pred[:, 1, :, :] > 0.5

            for k in range(computed, compute_end):
                output_data_writer.write_image_data(subdata_entries[k], pred_mask[k - computed, :, :].cpu().numpy().astype(np.uint8) * 255)

            if specific_inference:
                image_ground_truth_batch = torch.zeros((compute_end - computed, image_height, image_width),
                                                            dtype=torch.float32, device=config.device)
                image_crude_ground_truth_batch = torch.zeros((compute_end - computed, image_height, image_width),
                                                                  dtype=torch.float32, device=config.device)

                for k in range(computed, compute_end):
                    image_ground_truth_batch[k - computed, :, :] = torch.tensor(input_data_loader.get_segmentation_mask(subdata_entries[k], "blood_vessel"), dtype=torch.float32, device=config.device)
                    image_crude_ground_truth_batch[k - computed, :, :] = torch.tensor(input_data_loader.get_segmentation_mask(subdata_entries[k], "blood_vessel"), dtype=torch.float32, device=config.device)

                true_positive += int(
                    torch.sum((pred[:, 1, :, :] > 0.5) & (image_ground_truth_batch == 1)).item())
                true_negative += int(
                    torch.sum((pred[:, 1, :, :] <= 0.5) & (image_ground_truth_batch == 0)).item())
                false_positive += int(
                    torch.sum((pred[:, 1, :, :] > 0.5) & (image_ground_truth_batch == 0)).item())
                false_negative += int(
                    torch.sum((pred[:, 1, :, :] <= 0.5) & (image_ground_truth_batch == 1)).item())

                true_positive_crude += int(
                    torch.sum((pred[:, 0, :, :] > 0.5) & (image_crude_ground_truth_batch == 1)).item())
                true_negative_crude += int(
                    torch.sum((pred[:, 0, :, :] <= 0.5) & (image_crude_ground_truth_batch == 0)).item())
                false_positive_crude += int(
                    torch.sum((pred[:, 0, :, :] > 0.5) & (image_crude_ground_truth_batch == 0)).item())
                false_negative_crude += int(
                    torch.sum((pred[:, 0, :, :] <= 0.5) & (image_crude_ground_truth_batch == 1)).item())

        gc.collect()
        torch.cuda.empty_cache()

        computed = compute_end

        if computed - last_compute_print >= 100:
            print("Computed {} images in {:.2f} seconds".format(computed, time.time() - ctime))
            last_compute_print = computed
            ctime = time.time()

    if specific_inference:
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        accuracy_crude = (true_positive_crude + true_negative_crude) / (true_positive_crude + true_negative_crude + false_positive_crude + false_negative_crude)
        if true_positive_crude + false_positive_crude > 0:
            precision_crude = true_positive_crude / (true_positive_crude + false_positive_crude)
        else:
            precision_crude = 0
        if true_positive_crude + false_negative_crude > 0:
            recall_crude = true_positive_crude / (true_positive_crude + false_negative_crude)
        else:
            recall_crude = 0

        print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(accuracy, precision, recall))
        print("Accuracy Crude: {:.4f}, Precision Crude: {:.4f}, Recall Crude: {:.4f}".format(accuracy_crude, precision_crude, recall_crude))

    input_data_loader.close()
    output_data_writer.close()
    print("Inference Complete")
