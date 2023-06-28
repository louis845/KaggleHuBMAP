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
import inference_reconstructed_base
import obtain_reconstructed_binary_segmentation

def convert_predictions_to_image(pred_type: torch.Tensor):
    # pred_type is a tensor of size (batch_size, 512, 512) containing the predicted class for each pixel
    # there are num_classes + 1 possible classes, with 0 being the background class. Therefore the labels are [0, num_classes] inclusive.
    # Convert to a numpy array of size (batch_size, 512, 512, 3) representing the image for the predicted class, in RGB format.
    # The background class is black, and the other classes should have different hues.
    # The hue is determined by the class number, and the saturation and value are 255 constantly.
    # The conversion is done by converting the class number to a hue between 0 and 1, then converting that to an RGB value.

    pred_mask_image = np.zeros((compute_end - computed, 512, 512, 3), dtype=np.uint8)
    hue_mask = (255 * pred_type.to(torch.float32) / num_classes).cpu().numpy().astype(np.uint8)
    saturation_mask = ((pred_type > 0).to(torch.float32) * 255).cpu().numpy().astype(np.uint8)
    value_mask = saturation_mask

    pred_mask_image[:, :, :, 0] = hue_mask
    pred_mask_image[:, :, :, 1] = saturation_mask
    pred_mask_image[:, :, :, 2] = value_mask

    return pred_mask_image

def convert_confidence_levels_to_image(pred_type: torch.Tensor):
    pred_mask_image = np.zeros((compute_end - computed, 512, 512, 3), dtype=np.uint8)
    value_mask = (pred_type * 255).unsqueeze(-1).repeat(1, 1, 1, 3).cpu().numpy().astype(np.uint8)
    hue_mask = np.zeros_like(value_mask)
    saturation_mask = np.zeros_like(value_mask)

    pred_mask_image[:, :, :, 0] = hue_mask
    pred_mask_image[:, :, :, 1] = saturation_mask
    pred_mask_image[:, :, :, 2] = value_mask

    return pred_mask_image

def get_predictions(result: torch.Tensor, predictions_type:str="argmax"):
    if predictions_type == "argmax":
        return result.argmax(dim=1)
    elif predictions_type == "confidence":
        softmax = torch.softmax(result, dim=1)
        # (confidence level at least 0.5)
        return (softmax[:, 0, ...] <= 0.5) * (torch.argmax(result[:, 1:, ...], dim=1) + 1)
    elif predictions_type == "noconfidence":
        return torch.argmax(result[:, 1:, ...], dim=1) + 1
    else:
        softmax = torch.softmax(result, dim=1)
        return 1 - softmax[:, 0, ...]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of a multiclass U-Net model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use. Default 2.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true", help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--use_atrous_conv", action="store_true", help="Whether to use atrous convolutional networks. Default False.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")
    parser.add_argument("--unet_attention", action="store_true", help="Whether to use attention in the U-Net. Default False. Cannot be used with unet_plus.")
    parser.add_argument("--image_width", type=int, default=768, help="Width of the input images. Default 768.")
    parser.add_argument("--prediction_type", type=str, default="argmax", help="Type of prediction to use. Default argmax. Can be argmax, confidence, noconfidence, or levels.")

    model_data_manager.transform_add_argparse_arguments(parser)

    args = parser.parse_args()

    assert args.prediction_type in ["argmax", "confidence", "noconfidence", "levels"], "Invalid prediction type"

    input_data_loader, output_data_writer, model_path, subdata_entries, train_subdata_entries, val_subdata_entries = model_data_manager.transform_get_argparse_arguments(args)
    image_width = args.image_width
    classes = ["blood_vessel", "boundary"]
    class_weights = [1.0, 1.0]
    num_classes = 2

    gt_masks = {}
    for k in range(1, 3):
        gt_masks[k] = obtain_reconstructed_binary_segmentation.get_default_WSI_mask(k)

    if args.unet_attention:
        model = model_unet_attention.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                                    hidden_channels=args.hidden_channels,
                                                    use_batch_norm=args.use_batch_norm,
                                                    use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                    in_channels=4, use_atrous_conv=args.use_atrous_conv,
                                                    deep_supervision=True,
                                                    res_conv_blocks=[2, 3, 4, 6, 10, 10, 10]).to(device=config.device)
    else:
        model = model_unet_base.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                               hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                               use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                               in_channels=4, use_atrous_conv=args.use_atrous_conv,
                                               deep_supervision=True,
                                               res_conv_blocks=[2, 3, 4, 6, 10, 10, 10]).to(device=config.device)

    model_checkpoint_path = os.path.join(model_path, "model.pt")

    model.load_state_dict(torch.load(model_checkpoint_path))

    batch_size = args.batch_size

    computed = 0
    last_compute_print = 0
    ctime = time.time()

    if args.prediction_type != "levels":
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

    image_radius = image_width // 2
    while computed < len(subdata_entries):
        # Test the model
        with torch.no_grad():
            compute_end = min(computed + batch_size, len(subdata_entries))
            inference_batch = torch.zeros((compute_end - computed, 4, image_width, image_width), dtype=torch.float32, device=config.device)

            for k in range(computed, compute_end):
                inference_batch[k - computed, :, :, :] = inference_reconstructed_base.load_combined(subdata_entries[k], image_size=image_width)

            result, deep_outputs = model(inference_batch)
            del deep_outputs[:]
            # restrict result to center 512x512
            result = result[:, :, image_radius-256:image_radius+256, image_radius-256:image_radius+256]
            pred_type = get_predictions(result, args.prediction_type)
            if args.prediction_type == "levels":
                pred_mask_image = convert_confidence_levels_to_image(pred_type)
            else:
                pred_mask_image = convert_predictions_to_image(pred_type)

            for k in range(computed, compute_end):
                output_data_writer.write_image_data(subdata_entries[k],
                                                    cv2.cvtColor(pred_mask_image[k - computed, :, :, :], cv2.COLOR_HSV2RGB))

            if args.prediction_type != "levels":
                # Now we load the ground truth masks from the input data loader and compute the metrics
                # Obtain the ground truth labels
                ground_truth_batch = torch.zeros((compute_end - computed, 512, 512), dtype=torch.bool, device=config.device)
                ground_truth_class_labels_batch = torch.zeros((compute_end - computed, 512, 512), dtype=torch.long, device=config.device)
                for k in range(computed, compute_end):
                    seg_mask = input_data_loader.get_segmentation_mask(subdata_entries[k], "blood_vessel")
                    ground_truth_batch[k - computed, :, :] = torch.tensor(seg_mask, dtype=torch.bool, device=config.device)
                    x = model_data_manager.data_information.loc[subdata_entries[k], "i"]
                    y = model_data_manager.data_information.loc[subdata_entries[k], "j"]
                    wsi_id = model_data_manager.data_information.loc[subdata_entries[k], "source_wsi"]
                    ground_truth_class_labels_batch[k - computed, :, :] = torch.tensor(gt_masks[wsi_id].obtain_blood_vessel_mask(
                        x, x + 512, y, y + 512
                    ), dtype=torch.long, device=config.device)

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

    if args.prediction_type != "levels":
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