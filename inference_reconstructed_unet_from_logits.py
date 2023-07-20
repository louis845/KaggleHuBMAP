"""Inference of a model on a dataset. The model here is trained with reconstructed_model_progressive_supervised_unet.py"""

import gc
import os
import time
import argparse

import config

import cv2
import numpy as np
import tqdm

import torch
import torch.nn

import model_data_manager
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
    value_mask = (pred_type * 255).cpu().numpy().astype(np.uint8)
    hue_mask = np.zeros_like(value_mask)
    saturation_mask = np.zeros_like(value_mask)

    pred_mask_image[:, :, :, 0] = hue_mask
    pred_mask_image[:, :, :, 1] = saturation_mask
    pred_mask_image[:, :, :, 2] = value_mask

    return pred_mask_image


def get_predictions(result: torch.Tensor, predictions_type: str = "argmax"):
    if predictions_type == "argmax":
        return result.argmax(dim=1)
    elif predictions_type == "confidence":
        softmax = result
        # (confidence level at least 0.5)
        return (softmax[:, 0, ...] <= 0.5) * (torch.argmax(result[:, 1:, ...], dim=1) + 1)
    elif predictions_type == "noconfidence":
        return torch.argmax(result[:, 1:, ...], dim=1) + 1
    elif predictions_type == "levels":
        softmax = result
        return 1 - softmax[:, 0, ...]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of a multiclass U-Net model")
    parser.add_argument("--prediction_type", type=str, default="argmax",
                        help="Type of prediction to use. Default argmax. Can be argmax, confidence, noconfidence, levels or probas.")
    parser.add_argument("--reduction_logit_average", action="store_true",
                        help="Whether to average the logits. Default False.")
    parser.add_argument("--experts_only", action="store_true",
                        help="Whether to only use \"expert\" prediction. Default False.")
    parser.add_argument("--separated_logits", action="store_true", help="Whether to use separated logits. This is usually coupled with --use_separated_focal_loss in training script, and --use_separated_background in inference_reconstructed_logits.")
    parser.add_argument("--ignore_unknown", action="store_true", help="Whether to ignore unknown classes. Default False.")

    model_data_manager.transform_add_argparse_arguments(parser, requires_model=False)

    args = parser.parse_args()

    assert args.prediction_type in ["argmax", "confidence", "noconfidence", "levels", "probas"], "Invalid prediction type"

    input_data_loader, output_data_writer, model_path, subdata_entries, train_subdata_entries, val_subdata_entries = model_data_manager.transform_get_argparse_arguments(args, requires_model=False)

    assert input_data_loader.data_store is not None, "You must specify an input data which stores the logits, by the argument --original_data_name"

    classes = ["blood_vessel", "boundary"]
    class_weights = [1.0, 1.0]
    num_classes = 2

    gt_masks = {}
    for k in range(1, 3):
        gt_masks[k] = obtain_reconstructed_binary_segmentation.get_default_WSI_mask(k)

    computed = 0
    last_compute_print = 0
    ctime = time.time()

    if (args.prediction_type != "levels") and (args.prediction_type != "probas"):
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

    image_radius = 384
    logits_group = input_data_loader.data_store["logits"]  # type: h5py.Group
    reduction_logit_average = args.reduction_logit_average
    experts_only = args.experts_only
    separated_logits = args.separated_logits
    ignore_unknown = args.ignore_unknown
    print("Computing now. Prediction type: {}    Reduction logit average: {}    Experts only: {}    Separated logits: {}".format(args.prediction_type, reduction_logit_average, experts_only, separated_logits))
    with tqdm.tqdm(total=len(subdata_entries)) as pbar:
        while computed < len(subdata_entries):
            tile_id = subdata_entries[computed]
            compute_end = computed + 1
            # Get logits from computed input data
            with torch.no_grad():

                img_helper = inference_reconstructed_base.Composite1024To512ImageInference()
                img_helper.load_logits_from_hdf(logits_group, tile_id)
                result = img_helper.obtain_predictions(reduction_logit_average, experts_only, separated_logits).permute(2, 0, 1).unsqueeze(0)

                if args.prediction_type == "probas":
                    output_data_writer.write_image_data(tile_id, result.squeeze(0).permute(1, 2, 0).cpu().numpy())
                else:
                    pred_type = get_predictions(result, args.prediction_type)
                    if args.prediction_type == "levels":
                        pred_mask_image = convert_confidence_levels_to_image(pred_type)
                    else:
                        pred_mask_image = convert_predictions_to_image(pred_type)

                    for k in range(computed, compute_end):
                        output_data_writer.write_image_data(subdata_entries[k],
                                                            cv2.cvtColor(pred_mask_image[k - computed, :, :, :],
                                                                         cv2.COLOR_HSV2RGB))

                    if (args.prediction_type != "levels") and (args.prediction_type != "probas"):
                        # Now we load the ground truth masks from the input data loader and compute the metrics
                        # Obtain the ground truth labels
                        ground_truth_batch = torch.zeros((compute_end - computed, 512, 512), dtype=torch.bool, device=config.device)
                        ground_truth_class_labels_batch = torch.zeros((compute_end - computed, 512, 512), dtype=torch.long, device=config.device)
                        unknown_batch = torch.zeros((compute_end - computed, 512, 512), dtype=torch.bool, device=config.device)  # default no unknown
                        for k in range(computed, compute_end):
                            seg_mask = input_data_loader.get_segmentation_mask(subdata_entries[k], "blood_vessel")
                            ground_truth_batch[k - computed, :, :] = torch.tensor(seg_mask, dtype=torch.bool,
                                                                                  device=config.device)
                            x = model_data_manager.data_information.loc[subdata_entries[k], "i"]
                            y = model_data_manager.data_information.loc[subdata_entries[k], "j"]
                            wsi_id = model_data_manager.data_information.loc[subdata_entries[k], "source_wsi"]
                            ground_truth_class_labels_batch[k - computed, :, :] = torch.tensor(
                                gt_masks[wsi_id].obtain_blood_vessel_mask(
                                    x, x + 512, y, y + 512
                                ), dtype=torch.long, device=config.device)

                            if ignore_unknown:
                                unknown_mask = input_data_loader.get_segmentation_mask(subdata_entries[k], "unknown")
                                unknown_batch[k - computed, :, :] = torch.tensor(unknown_mask, dtype=torch.bool, device=config.device)

                    # Compute global metrics
                    true_positive += torch.sum((pred_type > 0) & ground_truth_batch & (~unknown_batch)).item()
                    true_negative += torch.sum((pred_type == 0) & ~ground_truth_batch & (~unknown_batch)).item()
                    false_positive += torch.sum((pred_type > 0) & ~ground_truth_batch & (~unknown_batch)).item()
                    false_negative += torch.sum((pred_type == 0) & ground_truth_batch & (~unknown_batch)).item()

                    for k in range(len(classes)):
                        seg_class = classes[k]
                        true_positive_classes[seg_class] += torch.sum(
                            (pred_type == (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & (~unknown_batch)).item()
                        true_negative_classes[seg_class] += torch.sum(
                            (pred_type != (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & (~unknown_batch)).item()
                        false_positive_classes[seg_class] += torch.sum(
                            (pred_type == (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & (~unknown_batch)).item()
                        false_negative_classes[seg_class] += torch.sum(
                            (pred_type != (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & (~unknown_batch)).item()

                    # Compute the train and test metrics
                    training_batch_mask = torch.zeros((compute_end - computed, 1, 1), dtype=torch.bool,
                                                      device=config.device)
                    validation_batch_mask = torch.zeros((compute_end - computed, 1, 1), dtype=torch.bool,
                                                        device=config.device)
                    for k in range(computed, compute_end):
                        if subdata_entries[k] in train_subdata_entries:
                            training_batch_mask[k - computed, 0, 0] = True
                        elif subdata_entries[k] in val_subdata_entries:
                            validation_batch_mask[k - computed, 0, 0] = True

                    true_positive_train += torch.sum((pred_type > 0) & ground_truth_batch & training_batch_mask & (~unknown_batch)).item()
                    true_negative_train += torch.sum(
                        (pred_type == 0) & ~ground_truth_batch & training_batch_mask & (~unknown_batch)).item()
                    false_positive_train += torch.sum(
                        (pred_type > 0) & ~ground_truth_batch & training_batch_mask & (~unknown_batch)).item()
                    false_negative_train += torch.sum(
                        (pred_type == 0) & ground_truth_batch & training_batch_mask & (~unknown_batch)).item()

                    true_positive_val += torch.sum((pred_type > 0) & ground_truth_batch & validation_batch_mask & (~unknown_batch)).item()
                    true_negative_val += torch.sum(
                        (pred_type == 0) & ~ground_truth_batch & validation_batch_mask & (~unknown_batch)).item()
                    false_positive_val += torch.sum(
                        (pred_type > 0) & ~ground_truth_batch & validation_batch_mask & (~unknown_batch)).item()
                    false_negative_val += torch.sum(
                        (pred_type == 0) & ground_truth_batch & validation_batch_mask & (~unknown_batch)).item()

                    for k in range(len(classes)):
                        seg_class = classes[k]
                        true_positive_classes_train[seg_class] += torch.sum((pred_type == (k + 1)) & (
                                    ground_truth_class_labels_batch == (k + 1)) & training_batch_mask & (~unknown_batch)).item()
                        true_negative_classes_train[seg_class] += torch.sum((pred_type != (k + 1)) & (
                                    ground_truth_class_labels_batch != (k + 1)) & training_batch_mask & (~unknown_batch)).item()
                        false_positive_classes_train[seg_class] += torch.sum((pred_type == (k + 1)) & (
                                    ground_truth_class_labels_batch != (k + 1)) & training_batch_mask & (~unknown_batch)).item()
                        false_negative_classes_train[seg_class] += torch.sum((pred_type != (k + 1)) & (
                                    ground_truth_class_labels_batch == (k + 1)) & training_batch_mask & (~unknown_batch)).item()

                        true_positive_classes_val[seg_class] += torch.sum((pred_type == (k + 1)) & (
                                    ground_truth_class_labels_batch == (k + 1)) & validation_batch_mask & (~unknown_batch)).item()
                        true_negative_classes_val[seg_class] += torch.sum((pred_type != (k + 1)) & (
                                    ground_truth_class_labels_batch != (k + 1)) & validation_batch_mask & (~unknown_batch)).item()
                        false_positive_classes_val[seg_class] += torch.sum((pred_type == (k + 1)) & (
                                    ground_truth_class_labels_batch != (k + 1)) & validation_batch_mask & (~unknown_batch)).item()
                        false_negative_classes_val[seg_class] += torch.sum((pred_type != (k + 1)) & (
                                    ground_truth_class_labels_batch == (k + 1)) & validation_batch_mask & (~unknown_batch)).item()

            gc.collect()
            torch.cuda.empty_cache()
            pbar.update(compute_end - computed)
            computed = compute_end

            if computed - last_compute_print >= 100:
                print("Computed {} images in {:.2f} seconds".format(computed, time.time() - ctime))
                last_compute_print = computed
                ctime = time.time()

    if (args.prediction_type != "levels") and (args.prediction_type != "probas"):
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        accuracy_train = (true_positive_train + true_negative_train) / (
                    true_positive_train + true_negative_train + false_positive_train + false_negative_train)
        accuracy_val = (true_positive_val + true_negative_val) / (
                    true_positive_val + true_negative_val + false_positive_val + false_negative_val)

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
            accuracy_classes[seg_class] = (true_positive_classes[seg_class] + true_negative_classes[seg_class]) / (
                        true_positive_classes[seg_class] + true_negative_classes[seg_class] + false_positive_classes[
                    seg_class] + false_negative_classes[seg_class])
            accuracy_classes_train[seg_class] = (true_positive_classes_train[seg_class] + true_negative_classes_train[
                seg_class]) / (true_positive_classes_train[seg_class] + true_negative_classes_train[seg_class] +
                               false_positive_classes_train[seg_class] + false_negative_classes_train[seg_class])
            accuracy_classes_val[seg_class] = (true_positive_classes_val[seg_class] + true_negative_classes_val[
                seg_class]) / (true_positive_classes_val[seg_class] + true_negative_classes_val[seg_class] +
                               false_positive_classes_val[seg_class] + false_negative_classes_val[seg_class])

            if true_positive_classes[seg_class] + false_positive_classes[seg_class] == 0:
                precision_classes[seg_class] = 0
            else:
                precision_classes[seg_class] = true_positive_classes[seg_class] / (
                            true_positive_classes[seg_class] + false_positive_classes[seg_class])
            if true_positive_classes_train[seg_class] + false_positive_classes_train[seg_class] == 0:
                precision_classes_train[seg_class] = 0
            else:
                precision_classes_train[seg_class] = true_positive_classes_train[seg_class] / (
                            true_positive_classes_train[seg_class] + false_positive_classes_train[seg_class])
            if true_positive_classes_val[seg_class] + false_positive_classes_val[seg_class] == 0:
                precision_classes_val[seg_class] = 0
            else:
                precision_classes_val[seg_class] = true_positive_classes_val[seg_class] / (
                            true_positive_classes_val[seg_class] + false_positive_classes_val[seg_class])

            if true_positive_classes[seg_class] + false_negative_classes[seg_class] == 0:
                recall_classes[seg_class] = 0
            else:
                recall_classes[seg_class] = true_positive_classes[seg_class] / (
                            true_positive_classes[seg_class] + false_negative_classes[seg_class])
            if true_positive_classes_train[seg_class] + false_negative_classes_train[seg_class] == 0:
                recall_classes_train[seg_class] = 0
            else:
                recall_classes_train[seg_class] = true_positive_classes_train[seg_class] / (
                            true_positive_classes_train[seg_class] + false_negative_classes_train[seg_class])
            if true_positive_classes_val[seg_class] + false_negative_classes_val[seg_class] == 0:
                recall_classes_val[seg_class] = 0
            else:
                recall_classes_val[seg_class] = true_positive_classes_val[seg_class] / (
                            true_positive_classes_val[seg_class] + false_negative_classes_val[seg_class])

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

        # Write the above prints to a file
        with open(os.path.join(output_data_writer.data_folder, "results.txt"), "w") as f:
            f.write("{:.4f} (Accuracy)\n".format(accuracy))
            f.write("{:.4f} (Precision)\n".format(precision))
            f.write("{:.4f} (Recall)\n".format(recall))
            f.write("{:.4f} (Accuracy Train)\n".format(accuracy_train))
            f.write("{:.4f} (Precision Train)\n".format(precision_train))
            f.write("{:.4f} (Recall Train)\n".format(recall_train))
            f.write("{:.4f} (Accuracy Val)\n".format(accuracy_val))
            f.write("{:.4f} (Precision Val)\n".format(precision_val))
            f.write("{:.4f} (Recall Val)\n".format(recall_val))
            for seg_class in classes:
                f.write("--------------- {} ---------------\n".format(seg_class))
                f.write("{:.4f} (Accuracy {})\n".format(accuracy_classes[seg_class], seg_class))
                f.write("{:.4f} (Precision {})\n".format(precision_classes[seg_class], seg_class))
                f.write("{:.4f} (Recall {})\n".format(recall_classes[seg_class], seg_class))
                f.write("{:.4f} (Accuracy {} Train)\n".format(accuracy_classes_train[seg_class], seg_class))
                f.write("{:.4f} (Precision {} Train)\n".format(precision_classes_train[seg_class], seg_class))
                f.write("{:.4f} (Recall {} Train)\n".format(recall_classes_train[seg_class], seg_class))
                f.write("{:.4f} (Accuracy {} Val)\n".format(accuracy_classes_val[seg_class], seg_class))
                f.write("{:.4f} (Precision {} Val)\n".format(precision_classes_val[seg_class], seg_class))
                f.write("{:.4f} (Recall {} Val)\n".format(recall_classes_val[seg_class], seg_class))

    input_data_loader.close()
    output_data_writer.close()
    print("Inference Complete")
