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
    value_mask = (pred_type * 255).cpu().numpy().astype(np.uint8)
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
        if use_separated_background:
            return (result[:, 0, ...] <= 0.0) * (torch.argmax(result[:, 1:, ...], dim=1) + 1)
        else:
            softmax = torch.softmax(result, dim=1)
            # (confidence level at least 0.5)
            return (softmax[:, 0, ...] <= 0.5) * (torch.argmax(result[:, 1:, ...], dim=1) + 1)
    elif predictions_type == "noconfidence":
        return torch.argmax(result[:, 1:, ...], dim=1) + 1
    else:
        softmax = torch.softmax(result, dim=1)
        return 1 - softmax[:, 0, ...]

def get_result_logits(model: torch.nn.Module, inference_batch: torch.Tensor, test_time_augmentation: bool):
    # inference_batch is a batch of images of shape (N, C, H, W)
    if test_time_augmentation:
        # obtain all variants using TTA
        result = get_result_logits(model, inference_batch, False)
        gc.collect()
        torch.cuda.empty_cache()
        result_rot90 = torch.rot90(get_result_logits(model, torch.rot90(inference_batch, 1, [2, 3]), False),
                                   -1, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()
        result_rot180 = torch.rot90(get_result_logits(model, torch.rot90(inference_batch, 2, [2, 3]), False),
                                    -2, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()
        result_rot270 = torch.rot90(get_result_logits(model, torch.rot90(inference_batch, 3, [2, 3]), False),
                                    -3, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()
        # flip
        result_flip = torch.flip(get_result_logits(model, torch.flip(inference_batch, [-1]), False), [-1])
        gc.collect()
        torch.cuda.empty_cache()
        result_rot90_flip = torch.rot90(torch.flip(get_result_logits(model, torch.flip(torch.rot90(inference_batch, 1, [2, 3]), [-1]), False),
                                                   [-1]), -1, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()
        result_rot180_flip = torch.rot90(torch.flip(get_result_logits(model, torch.flip(torch.rot90(inference_batch, 2, [2, 3]), [-1]), False),
                                                    [-1]), -2, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()
        result_rot270_flip = torch.rot90(torch.flip(get_result_logits(model, torch.flip(torch.rot90(inference_batch, 3, [2, 3]), [-1]), False),
                                                    [-1]), -3, [2, 3])
        gc.collect()
        torch.cuda.empty_cache()

        # average all results
        result = torch.mean(torch.stack([result, result_rot90, result_rot180, result_rot270,
                                         result_flip, result_rot90_flip, result_rot180_flip, result_rot270_flip], dim=0), dim=0)
        return result
    else:
        result, deep_outputs = model(inference_batch)
        del deep_outputs[:]
        # restrict result to center 512x512
        result = result[:, :, image_radius - 256:image_radius + 256, image_radius - 256:image_radius + 256]
        return result

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of a multiclass U-Net model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use. Default 2.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true", help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--use_atrous_conv", action="store_true", help="Whether to use atrous convolutional networks. Default False.")
    parser.add_argument("--use_squeeze_excitation", action="store_true", help="Whether to use squeeze and excitation. Default False.")
    parser.add_argument("--use_initial_conv", action="store_true", help="Whether to use the initial 7x7 kernel convolution. Default False.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[2, 3, 4, 6, 6, 7, 7], help="Number of hidden blocks for ResNets. Ignored if not resnet.")
    parser.add_argument("--use_tta", action="store_true", help="Whether to use test time augmentation. Default False.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--bottleneck_expansion", type=int, default=1, help="The expansion factor of the bottleneck. Default 1.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")
    parser.add_argument("--unet_attention", action="store_true", help="Whether to use attention in the U-Net. Default False. Cannot be used with unet_plus.")
    parser.add_argument("--image_width", type=int, default=768, help="Width of the input images. Default 768.")
    parser.add_argument("--prediction_type", type=str, default="argmax", help="Type of prediction to use. Default argmax. Can be argmax, confidence, noconfidence, or levels.")
    parser.add_argument("--use_separated_background", action="store_true",
                        help="Whether to use a separated outconv for background sigmoid. Default False. Must be used with atrous conv.")
    parser.add_argument("--ignore_unknown", action="store_true", help="Whether to ignore unknown classes. Default False.")

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

    blocks = args.hidden_blocks
    use_separated_background = args.use_separated_background
    if args.unet_attention:
        model = model_unet_attention.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                                    hidden_channels=args.hidden_channels,
                                                    use_batch_norm=args.use_batch_norm,
                                                    use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                    in_channels=4, use_atrous_conv=args.use_atrous_conv,
                                                    atrous_outconv_split=use_separated_background,
                                                    deep_supervision=True,
                                                    squeeze_excitation=args.use_squeeze_excitation,
                                                    bottleneck_expansion=args.bottleneck_expansion,
                                                    res_conv_blocks=blocks, use_initial_conv=args.use_initial_conv).to(device=config.device)
    else:
        model = model_unet_base.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                               hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                               use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                               deep_supervision=True,
                                               in_channels=4, use_atrous_conv=args.use_atrous_conv,
                                               atrous_outconv_split=use_separated_background,
                                               squeeze_excitation=args.use_squeeze_excitation,
                                               bottleneck_expansion=args.bottleneck_expansion,
                                               res_conv_blocks=blocks, use_initial_conv=args.use_initial_conv).to(device=config.device)

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
    print("Computing now. Prediction type: {}    Test time augmentation: {}".format(args.prediction_type, args.use_tta))
    print("Ignore unknown: {}".format(args.ignore_unknown))
    ignore_unknown = args.ignore_unknown
    with tqdm.tqdm(total=len(subdata_entries)) as pbar:
        while computed < len(subdata_entries):
            # Test the model
            with torch.no_grad():
                compute_end = min(computed + batch_size, len(subdata_entries))
                inference_batch = torch.zeros((compute_end - computed, 4, image_width, image_width), dtype=torch.float32, device=config.device)

                for k in range(computed, compute_end):
                    inference_batch[k - computed, :, :, :] = inference_reconstructed_base.load_combined(subdata_entries[k], image_size=image_width)
                
                result = get_result_logits(model, inference_batch, args.use_tta)
                pred_type = get_predictions(result, args.prediction_type)
                if args.prediction_type == "levels":
                    pred_mask_image = convert_confidence_levels_to_image(pred_type)
                else:
                    pred_mask_image = convert_predictions_to_image(pred_type)

                for k in range(computed, compute_end):
                    output_data_writer.write_image_data(subdata_entries[k],
                                                        cv2.cvtColor(pred_mask_image[k - computed, :, :, :], cv2.COLOR_HSV2RGB))
                    np.save(os.path.join(output_data_writer.data_folder, "{}_result".format(subdata_entries[k])), result[k - computed, :, :, :].detach().cpu().numpy())

                if args.prediction_type != "levels":
                    # Now we load the ground truth masks from the input data loader and compute the metrics
                    # Obtain the ground truth labels
                    ground_truth_batch = torch.zeros((compute_end - computed, 512, 512), dtype=torch.bool, device=config.device)
                    ground_truth_class_labels_batch = torch.zeros((compute_end - computed, 512, 512), dtype=torch.long, device=config.device)
                    unknown_batch = torch.zeros((compute_end - computed, 512, 512), dtype=torch.bool, device=config.device) # default no unknown
                    for k in range(computed, compute_end):
                        x = model_data_manager.data_information.loc[subdata_entries[k], "i"]
                        y = model_data_manager.data_information.loc[subdata_entries[k], "j"]
                        wsi_id = model_data_manager.data_information.loc[subdata_entries[k], "source_wsi"]

                        # seg_mask = gt_masks[wsi_id].obtain_blood_vessel_mask(x, x + 512, y, y + 512) > 0
                        seg_mask = input_data_loader.get_segmentation_mask(subdata_entries[k], "blood_vessel")
                        seg_mask = (cv2.dilate(seg_mask.astype(np.uint8) * 255, kernel, iterations=2) // 255).astype(bool)
                        ground_truth_batch[k - computed, :, :] = torch.tensor(seg_mask, dtype=torch.bool,
                                                                              device=config.device)
                        ground_truth_class_labels_batch[k - computed, :, :] = torch.tensor(gt_masks[wsi_id].obtain_blood_vessel_mask(
                            x, x + 512, y, y + 512
                        ), dtype=torch.long, device=config.device)

                        if ignore_unknown:
                            """unknown_mask = input_data_loader.get_segmentation_mask(subdata_entries[k], "unknown") # bool np array
                            # dilate the unknown mask by kernel with 2 iterations
                            unknown_mask = (cv2.dilate(unknown_mask.astype(np.uint8) * 255, kernel, iterations=2) // 255).astype(bool)"""
                            unknown_mask = gt_masks[wsi_id].obtain_unknown_mask(x, x + 512, y, y + 512) > 0
                            unknown_batch[k - computed, :, :] = torch.tensor(unknown_mask, dtype=torch.bool, device=config.device)

                    # Compute global metrics
                    true_positive += torch.sum((pred_type > 0) & ground_truth_batch & (~unknown_batch)).item()
                    true_negative += torch.sum((pred_type == 0) & ~ground_truth_batch & (~unknown_batch)).item()
                    false_positive += torch.sum((pred_type > 0) & ~ground_truth_batch & (~unknown_batch)).item()
                    false_negative += torch.sum((pred_type == 0) & ground_truth_batch & (~unknown_batch)).item()

                    for k in range(len(classes)):
                        seg_class = classes[k]
                        true_positive_classes[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & (~unknown_batch)).item()
                        true_negative_classes[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & (~unknown_batch)).item()
                        false_positive_classes[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & (~unknown_batch)).item()
                        false_negative_classes[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & (~unknown_batch)).item()

                    # Compute the train and test metrics
                    training_batch_mask = torch.zeros((compute_end - computed, 1, 1), dtype=torch.bool, device=config.device)
                    validation_batch_mask = torch.zeros((compute_end - computed, 1, 1), dtype=torch.bool, device=config.device)
                    for k in range(computed, compute_end):
                        if subdata_entries[k] in train_subdata_entries:
                            training_batch_mask[k - computed, 0, 0] = True
                        elif subdata_entries[k] in val_subdata_entries:
                            validation_batch_mask[k - computed, 0, 0] = True

                    true_positive_train += torch.sum((pred_type > 0) & ground_truth_batch & training_batch_mask & (~unknown_batch)).item()
                    true_negative_train += torch.sum((pred_type == 0) & ~ground_truth_batch & training_batch_mask & (~unknown_batch)).item()
                    false_positive_train += torch.sum((pred_type > 0) & ~ground_truth_batch & training_batch_mask & (~unknown_batch)).item()
                    false_negative_train += torch.sum((pred_type == 0) & ground_truth_batch & training_batch_mask & (~unknown_batch)).item()

                    true_positive_val += torch.sum((pred_type > 0) & ground_truth_batch & validation_batch_mask & (~unknown_batch)).item()
                    true_negative_val += torch.sum((pred_type == 0) & ~ground_truth_batch & validation_batch_mask & (~unknown_batch)).item()
                    false_positive_val += torch.sum((pred_type > 0) & ~ground_truth_batch & validation_batch_mask & (~unknown_batch)).item()
                    false_negative_val += torch.sum((pred_type == 0) & ground_truth_batch & validation_batch_mask & (~unknown_batch)).item()

                    for k in range(len(classes)):
                        seg_class = classes[k]
                        true_positive_classes_train[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & training_batch_mask & (~unknown_batch)).item()
                        true_negative_classes_train[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & training_batch_mask & (~unknown_batch)).item()
                        false_positive_classes_train[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & training_batch_mask & (~unknown_batch)).item()
                        false_negative_classes_train[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & training_batch_mask & (~unknown_batch)).item()

                        true_positive_classes_val[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & validation_batch_mask & (~unknown_batch)).item()
                        true_negative_classes_val[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & validation_batch_mask & (~unknown_batch)).item()
                        false_positive_classes_val[seg_class] += torch.sum((pred_type == (k + 1)) & (ground_truth_class_labels_batch != (k + 1)) & validation_batch_mask & (~unknown_batch)).item()
                        false_negative_classes_val[seg_class] += torch.sum((pred_type != (k + 1)) & (ground_truth_class_labels_batch == (k + 1)) & validation_batch_mask & (~unknown_batch)).item()

            gc.collect()
            torch.cuda.empty_cache()
            pbar.update(compute_end - computed)
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
