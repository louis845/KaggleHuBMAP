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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of a simple U-Net model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use. Default 2.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true", help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels to use. Default 3.")

    image_width = 512
    image_height = 512

    model_data_manager.transform_add_argparse_arguments(parser)

    args = parser.parse_args()

    input_data_loader, output_data_writer, model_path = model_data_manager.transform_get_argparse_arguments(args)


    model = model_unet_progressive.UNetClassifier(hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                           use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                           in_channels=args.in_channels).to(device=config.device)

    model_checkpoint_path = os.path.join(model_path, "model.pt")

    model.load_state_dict(torch.load(model_checkpoint_path))

    batch_size = args.batch_size

    computed = 0
    last_compute_print = 0
    ctime = time.time()
    while computed < len(model_data_manager.data_information):
        # Test the model
        with torch.no_grad():
            compute_end = min(computed + batch_size, len(model_data_manager.data_information))
            inference_batch = torch.zeros((compute_end - computed, 3, image_height, image_width), dtype=torch.float32, device=config.device)

            for k in range(computed, compute_end):
                inference_batch[k - computed, :, :, :] = torch.tensor(input_data_loader.get_image_data(model_data_manager.data_information.index[k]), dtype=torch.float32, device=config.device).permute(2, 0, 1)

            pred_mask = model(inference_batch)[:, 1, :, :] > 0.5

            for k in range(computed, compute_end):
                output_data_writer.write_image_data(model_data_manager.data_information.index[k], pred_mask[k - computed, :, :].cpu().numpy().astype(np.uint8) * 255)

        gc.collect()
        torch.cuda.empty_cache()

        computed = compute_end

        if computed - last_compute_print >= 100:
            print("Computed {} images in {:.2f} seconds".format(computed, time.time() - ctime))
            last_compute_print = computed
            ctime = time.time()


    input_data_loader.close()
    output_data_writer.close()
    print("Inference Complete")
