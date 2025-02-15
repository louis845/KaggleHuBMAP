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
import model_unet_attention
import model_multiclass_base

def refined_max(tensor: torch.Tensor):
    # compute the value of the maximum occurence along axis 1. First, compute whether positive values happen more often than zero values.
    binary = (tensor > 0).to(torch.long)
    binary = torch.mode(binary, dim=1)[0] # 0 if more zeros, 1 if more positive values

    # now compute the final mask.

def augmentation_rotation_flipping(model: torch.nn.Module, inference_batch: torch.Tensor, length: int, image_height: int, image_width: int, deep_supervision: bool):
    pred_types = torch.zeros((length, 8, image_height, image_width), dtype=torch.long,
                             device=config.device)
    for k in range(0, 360, 90):
        # rotate image by k degrees with torch functional
        inference_batch_rotate = torchvision.transforms.functional.rotate(inference_batch, k)
        if deep_supervision:
            result, deep_outputs = model(inference_batch_rotate)
            pred_type = torch.argmax(torchvision.transforms.functional.rotate(result, -k), dim=1)
            pred_types[:, k // 90, :, :] = pred_type
        else:
            result = model(inference_batch_rotate)
            pred_type = torch.argmax(torchvision.transforms.functional.rotate(result, -k), dim=1)
            pred_types[:, k // 90, :, :] = pred_type

    # flip image horizontally
    inference_batch = inference_batch.flip(3)
    for k in range(0, 360, 90):
        # rotate image by k degrees with torch functional
        inference_batch_rotate = torchvision.transforms.functional.rotate(inference_batch, k)
        if deep_supervision:
            result, deep_outputs = model(inference_batch_rotate)
            pred_type = torch.argmax(torchvision.transforms.functional.rotate(result, -k).flip(3), dim=1)
            pred_types[:, 4 + k // 90, :, :] = pred_type
        else:
            result = model(inference_batch_rotate)
            pred_type = torch.argmax(torchvision.transforms.functional.rotate(result, -k).flip(3), dim=1)
            pred_types[:, 4 + k // 90, :, :] = pred_type

    return pred_types

def augmentation_crop_rotation_flipping(model: torch.nn.Module, inference_batch: torch.Tensor, length: int, image_height: int, image_width: int,
                                   subimage_height: int, subimage_width: int, deep_supervision: bool):
    pred_type = torch.zeros((length, image_height, image_width), dtype=torch.long, device=config.device)

    top_left = torch.zeros((length, 8, image_height - subimage_height, image_width - subimage_width), dtype=torch.long, device=config.device)
    top_right = torch.zeros((length, 8, image_height - subimage_height, image_width - subimage_width), dtype=torch.long, device=config.device)
    bottom_left = torch.zeros((length, 8, image_height - subimage_height, image_width - subimage_width), dtype=torch.long, device=config.device)
    bottom_right = torch.zeros((length, 8, image_height - subimage_height, image_width - subimage_width), dtype=torch.long, device=config.device)

    top = torch.zeros((length, 16, image_height - subimage_height, 2 * subimage_width - image_width), dtype=torch.long, device=config.device)
    bottom = torch.zeros((length, 16, image_height - subimage_height, 2 * subimage_width - image_width), dtype=torch.long, device=config.device)
    left = torch.zeros((length, 16, 2 * subimage_height - image_height, image_width - subimage_width), dtype=torch.long, device=config.device)
    right = torch.zeros((length, 16, 2 * subimage_height - image_height, image_width - subimage_width), dtype=torch.long, device=config.device)

    center = torch.zeros((length, 32, 2 * subimage_height - image_height, 2 * subimage_width - image_width), dtype=torch.long, device=config.device)


    top_left_pred_types = augmentation_rotation_flipping(model, inference_batch[:, :, :subimage_height, :subimage_width], length, subimage_height, subimage_width, deep_supervision)
    bottom_left_pred_types = augmentation_rotation_flipping(model, inference_batch[:, :, -subimage_height:, :subimage_width], length, subimage_height, subimage_width, deep_supervision)
    top_right_pred_types = augmentation_rotation_flipping(model, inference_batch[:, :, :subimage_height, -subimage_width:], length, subimage_height, subimage_width, deep_supervision)
    bottom_right_pred_types = augmentation_rotation_flipping(model, inference_batch[:, :, -subimage_height:, -subimage_width:], length, subimage_height, subimage_width, deep_supervision)

    # Fill in the predictions
    top_left[:, :, :, :] = top_left_pred_types[:, :, :image_height - subimage_height, :image_width - subimage_width]
    bottom_left[:, :, :, :] = bottom_left_pred_types[:, :, subimage_height - image_height:, :image_width - subimage_width]
    top_right[:, :, :, :] = top_right_pred_types[:, :, :image_height - subimage_height, subimage_width - image_width:]
    bottom_right[:, :, :, :] = bottom_right_pred_types[:, :, subimage_height - image_height:, subimage_width - image_width:]

    top[:, :8, :, :] = top_left_pred_types[:, :, :image_height - subimage_height, image_width - subimage_width:]
    top[:, 8:, :, :] = top_right_pred_types[:, :, :image_height - subimage_height, :subimage_width - image_width]
    bottom[:, :8, :, :] = bottom_left_pred_types[:, :, subimage_height - image_height:, image_width - subimage_width:]
    bottom[:, 8:, :, :] = bottom_right_pred_types[:, :, subimage_height - image_height:, :subimage_width - image_width]

    left[:, :8, :, :] = top_left_pred_types[:, :, image_height - subimage_height:, :image_width - subimage_width]
    left[:, 8:, :, :] = bottom_left_pred_types[:, :, :subimage_height - image_height, :image_width - subimage_width]
    right[:, :8, :, :] = top_right_pred_types[:, :, image_height - subimage_height:, subimage_width - image_width:]
    right[:, 8:, :, :] = bottom_right_pred_types[:, :, :subimage_height - image_height, subimage_width - image_width:]

    center[:, :8, :, :] = top_left_pred_types[:, :, image_height - subimage_height:, image_width - subimage_width:]
    center[:, 8:16, :, :] = top_right_pred_types[:, :, image_height - subimage_height:, :subimage_width - image_width]
    center[:, 16:24, :, :] = bottom_left_pred_types[:, :, :subimage_height - image_height, image_width - subimage_width:]
    center[:, 24:, :, :] = bottom_right_pred_types[:, :, :subimage_height - image_height, :subimage_width - image_width]

    pred_type[:, :-subimage_height, :-subimage_width] = torch.mode(top_left, dim=1)[0]
    pred_type[:, :-subimage_height, subimage_width:] = torch.mode(top_right, dim=1)[0]
    pred_type[:, subimage_height:, :-subimage_width] = torch.mode(bottom_left, dim=1)[0]
    pred_type[:, subimage_height:, subimage_width:] = torch.mode(bottom_right, dim=1)[0]

    pred_type[:, :-subimage_height, -subimage_width:subimage_width] = torch.mode(top, dim=1)[0]
    pred_type[:, subimage_height:, -subimage_width:subimage_width] = torch.mode(bottom, dim=1)[0]
    pred_type[:, -subimage_height:subimage_height, :-subimage_width] = torch.mode(left, dim=1)[0]
    pred_type[:, -subimage_height:subimage_height, subimage_width:] = torch.mode(right, dim=1)[0]

    pred_type[:, -subimage_height:subimage_height, -subimage_width:subimage_width] = torch.mode(center, dim=1)[0]

    return pred_type


def augmentation_rotation_flipping_result(model: torch.nn.Module, inference_batch: torch.Tensor, length: int, image_height: int, image_width: int, deep_supervision: bool):
    pred_types = augmentation_rotation_flipping(model, inference_batch, length, image_height, image_width, deep_supervision)

    pred_type, _ = torch.max(pred_types, dim=1)
    return pred_type

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
    parser.add_argument("--use_augmentation", action="store_true", help="Whether to use multiple augmentations and majority voting for inference. Default False.")

    image_width = 512
    image_height = 512

    model_data_manager.transform_add_argparse_arguments(parser)

    args = parser.parse_args()

    input_data_loader, output_data_writer, model_path, subdata_entries, train_subdata_entries, val_subdata_entries = model_data_manager.transform_get_argparse_arguments(args)

    deep_supervision = args.deep_supervision
    deep_supervision_levels = args.deep_supervision_levels
    assert (deep_supervision_levels == 0) or deep_supervision, "Cannot use deep supervision levels without deep supervision"

    use_augmentation = args.use_augmentation

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
    entries_masks = model_multiclass_base.precompute_classes(input_data_loader, subdata_entries, classes)

    batch_size = args.batch_size

    computed = 0
    last_compute_print = 0
    ctime = time.time()

    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    true_positive_classes, true_negative_classes, false_positive_classes, false_negative_classes = {}, {}, {}, {}
    true_positive_train, true_negative_train, false_positive_train, false_negative_train = 0, 0, 0, 0
    true_positive_classes_train, true_negative_classes_train, false_positive_classes_train, false_negative_classes_train = {}, {}, {}, {}
    true_positive_val, true_negative_val, false_positive_val, false_negative_val = 0, 0, 0, 0
    true_positive_classes_val, true_negative_classes_val, false_positive_classes_val, false_negative_classes_val = {}, {}, {}, {}

    for seg_class in classes:
        true_positive_classes[seg_class] = 0.0
        true_negative_classes[seg_class] = 0.0
        false_positive_classes[seg_class] = 0.0
        false_negative_classes[seg_class] = 0.0

        true_positive_classes_train[seg_class] = 0.0
        true_negative_classes_train[seg_class] = 0.0
        false_positive_classes_train[seg_class] = 0.0
        false_negative_classes_train[seg_class] = 0.0

        true_positive_classes_val[seg_class] = 0.0
        true_negative_classes_val[seg_class] = 0.0
        false_positive_classes_val[seg_class] = 0.0
        false_negative_classes_val[seg_class] = 0.0

    while computed < len(subdata_entries):
        # Test the model
        with torch.no_grad():
            compute_end = min(computed + batch_size, len(subdata_entries))
            inference_batch = torch.zeros((compute_end - computed, 3, image_height, image_width), dtype=torch.float32, device=config.device)

            for k in range(computed, compute_end):
                inference_batch[k - computed, :, :, :] = torch.tensor(input_data_loader.get_image_data(subdata_entries[k]), dtype=torch.float32, device=config.device).permute(2, 0, 1)

            if use_augmentation:
                #pred_type = augmentation_rotation_flipping_result(model, inference_batch, compute_end - computed, image_height, image_width, deep_supervision)
                pred_type = augmentation_crop_rotation_flipping(model, inference_batch, compute_end - computed, image_height, image_width, 448, 448, deep_supervision)
            else:
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


            for k in range(computed, compute_end):
                output_data_writer.write_image_data(subdata_entries[k],
                                                    cv2.cvtColor(pred_mask_image[k - computed, :, :, :], cv2.COLOR_HSV2RGB))

            # Now we load the ground truth masks from the input data loader and compute the metrics
            # Obtain the ground truth labels
            ground_truth_batch = torch.zeros((compute_end - computed, image_height, image_width), dtype=torch.bool, device=config.device)
            ground_truth_class_labels_batch = torch.zeros((compute_end - computed, image_height, image_width), dtype=torch.long, device=config.device)
            for k in range(computed, compute_end):
                seg_mask = input_data_loader.get_segmentation_mask(subdata_entries[k], "blood_vessel")
                ground_truth_batch[k - computed, :, :] = torch.tensor(seg_mask, dtype=torch.bool, device=config.device)
                ground_truth_class_labels_batch[k - computed, :, :] = torch.tensor(entries_masks[subdata_entries[k]], dtype=torch.long, device=config.device)

            # Compute global metrics
            true_positive += torch.sum((pred_type > 0) & ground_truth_batch).item()
            true_negative += torch.sum((pred_type == 0) & ~ground_truth_batch).item()
            false_positive += torch.sum((pred_type > 0) & ~ground_truth_batch).item()
            false_negative += torch.sum((pred_type == 0) & ground_truth_batch).item()

            for k in range(len(classes)):
                seg_class = classes[k]
                true_positive_classes[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch == (k + 1))).item()
                true_negative_classes[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch != (k + 1))).item()
                false_positive_classes[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch != (k + 1))).item()
                false_negative_classes[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch == (k + 1))).item()

            # Compute the train and test metrics
            training_batch_mask = torch.zeros((compute_end - computed, 1, 1), dtype=torch.bool, device=config.device)
            validation_batch_mask = torch.zeros((compute_end - computed, 1, 1), dtype=torch.bool, device=config.device)
            for k in range(computed, compute_end):
                if subdata_entries[k] in train_subdata_entries:
                    training_batch_mask[k - computed, 0, 0] = True
                elif subdata_entries[k] in val_subdata_entries:
                    validation_batch_mask[k - computed, 0, 0] = True

            true_positive_train += torch.sum((pred_type > 0) & ground_truth_batch & training_batch_mask).item()
            true_negative_train += torch.sum((pred_type == 0) & ~ground_truth_batch & training_batch_mask).item()
            false_positive_train += torch.sum((pred_type > 0) & ~ground_truth_batch & training_batch_mask).item()
            false_negative_train += torch.sum((pred_type == 0) & ground_truth_batch & training_batch_mask).item()

            true_positive_val += torch.sum((pred_type > 0) & ground_truth_batch & validation_batch_mask).item()
            true_negative_val += torch.sum((pred_type == 0) & ~ground_truth_batch & validation_batch_mask).item()
            false_positive_val += torch.sum((pred_type > 0) & ~ground_truth_batch & validation_batch_mask).item()
            false_negative_val += torch.sum((pred_type == 0) & ground_truth_batch & validation_batch_mask).item()

            for k in range(len(classes)):
                seg_class = classes[k]
                true_positive_classes_train[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & training_batch_mask).item()
                true_negative_classes_train[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & training_batch_mask).item()
                false_positive_classes_train[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & training_batch_mask).item()
                false_negative_classes_train[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & training_batch_mask).item()

                true_positive_classes_val[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & validation_batch_mask).item()
                true_negative_classes_val[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & validation_batch_mask).item()
                false_positive_classes_val[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & validation_batch_mask).item()
                false_negative_classes_val[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & validation_batch_mask).item()

        gc.collect()
        torch.cuda.empty_cache()

        computed = compute_end

        if computed - last_compute_print >= 100:
            print("Computed {} images in {:.2f} seconds".format(computed, time.time() - ctime))
            last_compute_print = computed
            ctime = time.time()

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    accuracy_train = (true_positive_train + true_negative_train) / (true_positive_train + true_negative_train + false_positive_train + false_negative_train)
    accuracy_val = (true_positive_val + true_negative_val) / (true_positive_val + true_negative_val + false_positive_val + false_negative_val)

    if true_positive + false_positive == 0:
        precision = 0
    else:
        precision = true_positive / (true_positive + false_positive)
    if true_positive_train + false_positive_train == 0:
        precision_train = 0
    else:
        precision_train = true_positive_train / (true_positive_train + false_positive_train)
    if true_positive_val + false_positive_val == 0:
        precision_val = 0
    else:
        precision_val = true_positive_val / (true_positive_val + false_positive_val)

    if true_positive + false_negative == 0:
        recall = 0
    else:
        recall = true_positive / (true_positive + false_negative)
    if true_positive_train + false_negative_train == 0:
        recall_train = 0
    else:
        recall_train = true_positive_train / (true_positive_train + false_negative_train)
    if true_positive_val + false_negative_val == 0:
        recall_val = 0
    else:
        recall_val = true_positive_val / (true_positive_val + false_negative_val)

    accuracy_classes = {}
    precision_classes = {}
    recall_classes = {}
    accuracy_classes_train = {}
    precision_classes_train = {}
    recall_classes_train = {}
    accuracy_classes_val = {}
    precision_classes_val = {}
    recall_classes_val = {}

    for seg_class in classes:
        accuracy_classes[seg_class] = (true_positive_classes[seg_class] + true_negative_classes[seg_class]) / (true_positive_classes[seg_class] + true_negative_classes[seg_class] + false_positive_classes[seg_class] + false_negative_classes[seg_class])
        accuracy_classes_train[seg_class] = (true_positive_classes_train[seg_class] + true_negative_classes_train[seg_class]) / (true_positive_classes_train[seg_class] + true_negative_classes_train[seg_class] + false_positive_classes_train[seg_class] + false_negative_classes_train[seg_class])
        accuracy_classes_val[seg_class] = (true_positive_classes_val[seg_class] + true_negative_classes_val[seg_class]) / (true_positive_classes_val[seg_class] + true_negative_classes_val[seg_class] + false_positive_classes_val[seg_class] + false_negative_classes_val[seg_class])

        if true_positive_classes[seg_class] + false_positive_classes[seg_class] == 0:
            precision_classes[seg_class] = 0
        else:
            precision_classes[seg_class] = true_positive_classes[seg_class] / (true_positive_classes[seg_class] + false_positive_classes[seg_class])
        if true_positive_classes_train[seg_class] + false_positive_classes_train[seg_class] == 0:
            precision_classes_train[seg_class] = 0
        else:
            precision_classes_train[seg_class] = true_positive_classes_train[seg_class] / (true_positive_classes_train[seg_class] + false_positive_classes_train[seg_class])
        if true_positive_classes_val[seg_class] + false_positive_classes_val[seg_class] == 0:
            precision_classes_val[seg_class] = 0
        else:
            precision_classes_val[seg_class] = true_positive_classes_val[seg_class] / (true_positive_classes_val[seg_class] + false_positive_classes_val[seg_class])

        if true_positive_classes[seg_class] + false_negative_classes[seg_class] == 0:
            recall_classes[seg_class] = 0
        else:
            recall_classes[seg_class] = true_positive_classes[seg_class] / (true_positive_classes[seg_class] + false_negative_classes[seg_class])
        if true_positive_classes_train[seg_class] + false_negative_classes_train[seg_class] == 0:
            recall_classes_train[seg_class] = 0
        else:
            recall_classes_train[seg_class] = true_positive_classes_train[seg_class] / (true_positive_classes_train[seg_class] + false_negative_classes_train[seg_class])
        if true_positive_classes_val[seg_class] + false_negative_classes_val[seg_class] == 0:
            recall_classes_val[seg_class] = 0
        else:
            recall_classes_val[seg_class] = true_positive_classes_val[seg_class] / (true_positive_classes_val[seg_class] + false_negative_classes_val[seg_class])

    print("{:.4f} (Accuracy)".format(accuracy))
    print("{:.4f} (Precision)".format(precision))
    print("{:.4f} (Recall)".format(recall))
    print("{:.4f} (Accuracy Train)".format(accuracy_train))
    print("{:.4f} (Precision Train)".format(precision_train))
    print("{:.4f} (Recall Train)".format(recall_train))
    print("{:.4f} (Accuracy Val)".format(accuracy_val))
    print("{:.4f} (Precision Val)".format(precision_val))
    print("{:.4f} (Recall Val)".format(recall_val))
    for seg_class in classes:
        print("--------------- {} ---------------".format(seg_class))
        print("{:.4f} (Accuracy {})".format(accuracy_classes[seg_class], seg_class))
        print("{:.4f} (Precision {})".format(precision_classes[seg_class], seg_class))
        print("{:.4f} (Recall {})".format(recall_classes[seg_class], seg_class))
        print("{:.4f} (Accuracy {} Train)".format(accuracy_classes_train[seg_class], seg_class))
        print("{:.4f} (Precision {} Train)".format(precision_classes_train[seg_class], seg_class))
        print("{:.4f} (Recall {} Train)".format(recall_classes_train[seg_class], seg_class))
        print("{:.4f} (Accuracy {} Val)".format(accuracy_classes_val[seg_class], seg_class))
        print("{:.4f} (Precision {} Val)".format(precision_classes_val[seg_class], seg_class))
        print("{:.4f} (Recall {} Val)".format(recall_classes_val[seg_class], seg_class))


    input_data_loader.close()
    output_data_writer.close()
    print("Inference Complete")
