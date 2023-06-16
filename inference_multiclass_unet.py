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
import model_unet_base
import model_unet_plus
import model_unet_attention
import model_multiclass_base

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of a multiclass U-Net model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use. Default 2.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true", help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--use_atrous_conv", action="store_true", help="Whether to use atrous convolutional networks. Default False.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")
    parser.add_argument("--unet_attention", action="store_true", help="Whether to use attention in the U-Net. Default False. Cannot be used with unet_plus.")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels to use. Default 3.")
    parser.add_argument("--multiclass_config", type=str, default="multiclass_config", help="Path to the multiclass config file for multiple class definition. Default None.")
    parser.add_argument("--deep_supervision", action="store_true", help="Whether the model is trained with deep supervision. Default False.")
    parser.add_argument("--deep_supervision_levels", type=int, default=0, help="How many multiclass deep supervision levels to use. Default 0.")

    image_width = 512
    image_height = 512

    model_data_manager.transform_add_argparse_arguments(parser)

    args = parser.parse_args()

    input_data_loader, output_data_writer, model_path, subdata_entries = model_data_manager.transform_get_argparse_arguments(args)

    deep_supervision = args.deep_supervision
    deep_supervision_levels = args.deep_supervision_levels
    assert (deep_supervision_levels == 0) or deep_supervision, "Cannot use deep supervision levels without deep supervision"

    if not os.path.exists("{}.json".format(args.multiclass_config)):
        print("Multiclass config file {}.json does not exist".format(args.multiclass_config))
        exit(-1)
    multiclass_config, classes, num_classes = model_multiclass_base.load_multiclass_config("{}.json".format(args.multiclass_config))

    if deep_supervision:
        if args.unet_attention:
            model = model_unet_attention.UNetClassifier(num_classes=num_classes, hidden_channels=args.hidden_channels,
                                                        use_batch_norm=args.use_batch_norm,
                                                        use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                        in_channels=args.in_channels, use_atrous_conv=args.use_atrous_conv,
                                                        deep_supervision=True, num_deep_multiclasses=deep_supervision_levels).to(device=config.device)
        else:
            model = model_unet_base.UNetClassifier(num_classes=num_classes, hidden_channels=args.hidden_channels,
                                                   use_batch_norm=args.use_batch_norm,
                                                   use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                   in_channels=args.in_channels, use_atrous_conv=args.use_atrous_conv,
                                                   deep_supervision=True, num_deep_multiclasses=deep_supervision_levels).to(device=config.device)
    else:
        if args.unet_attention:
            model = model_unet_attention.UNetClassifier(num_classes=num_classes, hidden_channels=args.hidden_channels,
                                                        use_batch_norm=args.use_batch_norm,
                                                        use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                        in_channels=args.in_channels, use_atrous_conv=args.use_atrous_conv,
                                                        deep_supervision=False).to(device=config.device)
        else:
            model = model_unet_base.UNetClassifier(num_classes=num_classes, hidden_channels=args.hidden_channels,
                                                   use_batch_norm=args.use_batch_norm,
                                                   use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                   in_channels=args.in_channels, use_atrous_conv=args.use_atrous_conv,
                                                   deep_supervision=False).to(device=config.device)

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

            if deep_supervision:
                result, deep_outputs = model(inference_batch)
                pred_type = torch.argmax(result, dim=1)
            else:
                result = model(inference_batch)
                pred_type = torch.argmax(result, dim=1)

            # pred_type is a tensor of size (batch_size, image_height, image_width) containing the predicted class for each pixel
            # there are num_classes + 1 possible classes, with 0 being the background class. Therefore the labels are [0, num_classes] inclusive.
            # Convert to a numpy array of size (batch_size, image_height, image_width, 3) representing the image for the predicted class, in RGB format.
            # The background class is black, and the other classes should have different hues.
            # The hue is determined by the class number, and the saturation and value are 255 constantly.
            # The conversion is done by converting the class number to a hue between 0 and 1, then converting that to an RGB value.

            pred_mask_image = np.zeros((compute_end - computed, image_height, image_width, 3), dtype=np.uint8)
            hue_mask = (255 * pred_type.to(torch.float32) / num_classes).cpu().numpy().astype(np.uint8)
            saturation_mask = ((pred_type > 0).to(torch.float32) * 255).cpu().numpy().astype(np.uint8)
            value_mask = saturation_mask

            pred_mask_image[:, :, :, 0] = hue_mask
            pred_mask_image[:, :, :, 1] = saturation_mask
            pred_mask_image[:, :, :, 2] = value_mask

            pred_mask_image = cv2.cvtColor(pred_mask_image, cv2.COLOR_HSV2RGB)

            for k in range(computed, compute_end):
                output_data_writer.write_image_data(model_data_manager.data_information.index[k], pred_mask_image[k - computed, :, :, :])

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
