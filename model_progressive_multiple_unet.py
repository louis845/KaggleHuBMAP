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
import torchvision.transforms.functional

import model_data_manager
import model_unet_progressive
import model_simple_unet

def convert_train_history(train_history, out_probas, omit_train_probas):
    mod_history = {
        "loss_cum": train_history["loss_cum"],
        "val_loss_cum": train_history["val_loss_cum"]
    }

    for k in range(omit_train_probas, out_probas):
        mod_history["loss_{}".format(k)] = []
        mod_history["val_loss_{}".format(k)] = []
        mod_history["accuracy_{}".format(k)] = []
        mod_history["val_accuracy_{}".format(k)] = []
        mod_history["precision_{}".format(k)] = []
        mod_history["val_precision_{}".format(k)] = []
        mod_history["recall_{}".format(k)] = []
        mod_history["val_recall_{}".format(k)] = []

        for j in range(len(train_history["loss_cum"])):
            mod_history["loss_{}".format(k)].append(train_history["loss"][j][k - omit_train_probas])
            mod_history["val_loss_{}".format(k)].append(train_history["val_loss"][j][k - omit_train_probas])
            mod_history["accuracy_{}".format(k)].append(train_history["accuracy"][j][k - omit_train_probas])
            mod_history["val_accuracy_{}".format(k)].append(train_history["val_accuracy"][j][k - omit_train_probas])
            mod_history["precision_{}".format(k)].append(train_history["precision"][j][k - omit_train_probas])
            mod_history["val_precision_{}".format(k)].append(train_history["val_precision"][j][k - omit_train_probas])
            mod_history["recall_{}".format(k)].append(train_history["recall"][j][k - omit_train_probas])
            mod_history["val_recall_{}".format(k)].append(train_history["val_recall"][j][k - omit_train_probas])

    return mod_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple U-Net model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--rotation_augmentation", action="store_true", help="Whether to use rotation augmentation. Default False.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use. Default 2.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate to use. Default 1e-5.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true", help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--use_atrous_conv", action="store_true", help="Whether to use atrous convolutional networks. Default False.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")
    parser.add_argument("--unet_attention", action="store_true", help="Whether to use attention in the U-Net. Default False. Cannot be used with unet_plus.")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels to use. Default 3.")
    parser.add_argument("--out_probas", type=int, default=2, help="Number of output probabilities to use. Default 2.")
    parser.add_argument("--omit_train_probas", type=int, default=0, help="Number of output probabilities to omit from training. Default 0.")
    parser.add_argument("--background", type=str, default="blood_vessel", help="The background mask to be used. This is for determining the weights. Default blood_vessel.")
    parser.add_argument("--ground_truth", type=str, default="blood_vessel", help="The segmentation mask to be used for ground truth. Default blood_vessel.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="The number of epochs to wait before stopping training. Default 10.")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.01, help="The threshold to use for early stopping. Default 0.01.")
    parser.add_argument("--gradient_momentum_multiplication", action="store_true", help="Whether to use gradient momentum multiplication. Default False.")

    image_width = 512
    image_height = 512

    model_data_manager.model_add_argparse_arguments(parser)

    args = parser.parse_args()

    model_dir, dataset_loader, training_entries, validation_entries, prev_model_checkpoint_dir, extra_info = model_data_manager.model_get_argparse_arguments(args)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    training_entries = np.array(training_entries, dtype=object)
    validation_entries = np.array(validation_entries, dtype=object)

    model = model_unet_progressive.UNetProgressiveWrapper(training=args.learning_rate, hidden_channels=args.hidden_channels, pyr_height=args.pyramid_height,
                                                          in_channels=args.in_channels, use_batch_norm=args.use_batch_norm, use_res_conv=args.use_res_conv,
                                                          use_atrous_conv=args.use_atrous_conv, unet_attention=args.unet_attention, outputs=args.out_probas)

    if prev_model_checkpoint_dir is not None:
        model.load_model(prev_model_checkpoint_dir)
        model.set_learning_rates(args.learning_rate)
        if args.gradient_momentum_multiplication:
            model.multiply_gradient_momentums()


    batch_size = args.batch_size
    num_epochs = args.epochs
    rotation_augmentation = args.rotation_augmentation
    epochs_per_save = args.epochs_per_save
    image_pixels_round = 2 ** args.pyramid_height
    in_channels = args.in_channels
    background = args.background
    ground_truth = args.ground_truth
    out_probas = args.out_probas
    omit_train_probas = args.omit_train_probas

    if omit_train_probas > out_probas:
        print("Cannot omit more probabilities than there are. omit_train_probas: {}, out_probas: {}".format(omit_train_probas, out_probas))
        exit(1)

    model_config = {
        "model": "model_progressive_multiple_unet",
        "epochs": num_epochs,
        "rotation_augmentation": rotation_augmentation,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "epochs_per_save": epochs_per_save,
        "use_batch_norm": args.use_batch_norm,
        "use_res_conv": args.use_res_conv,
        "use_atrous_conv": args.use_atrous_conv,
        "hidden_channels": args.hidden_channels,
        "pyramid_height": args.pyramid_height,
        "unet_attention": args.unet_attention,
        "in_channels": args.in_channels,
        "out_probas": args.out_probas,
        "omit_train_probas": args.omit_train_probas,
        "background": args.background,
        "ground_truth": args.ground_truth,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "gradient_momentum_multiplication": args.gradient_momentum_multiplication,
        "training_script": "model_progressive_multiple_unet.py",
    }
    for key, value in extra_info.items():
        model_config[key] = value

    # Compute the number of positive and negative pixels in the training data
    if background not in dataset_loader.list_segmentation_masks():
        print("Invalid background weights split. The available options are: {}".format(dataset_loader.list_segmentation_masks()))
        exit(1)
    if ground_truth not in dataset_loader.list_segmentation_masks():
        print("Invalid ground truth. The available options are: {}".format(dataset_loader.list_segmentation_masks()))
        exit(1)

    with torch.no_grad():
        num_background_positive_pixels, num_background_negative_pixels, num_foreground_positive_pixels,\
            num_foreground_negative_pixels, num_dset1_entries = model_simple_unet.compute_background_foreground(training_entries, dataset_loader, ground_truth, background)

        print("Number of foreground positive pixels: {}".format(num_foreground_positive_pixels))
        print("Number of foreground negative pixels: {}".format(num_foreground_negative_pixels))
        print("Number of background positive pixels: {}".format(num_background_positive_pixels))
        print("Number of background negative pixels: {}".format(num_background_negative_pixels))

        foreground_weight = (num_background_negative_pixels - num_background_positive_pixels) / (num_foreground_positive_pixels - num_foreground_negative_pixels + num_background_negative_pixels - num_background_positive_pixels)
        background_weight = (num_foreground_positive_pixels - num_foreground_negative_pixels) / (num_foreground_positive_pixels - num_foreground_negative_pixels + num_background_negative_pixels - num_background_positive_pixels)

        print("Foreground weight: {}".format(foreground_weight))
        print("Background weight: {}".format(background_weight))

    rng = np.random.default_rng()

    train_history = {"loss_cum": [], "loss": [], "accuracy": [], "precision": [], "recall": [],
                     "val_loss_cum": [], "val_loss": [], "val_accuracy": [], "val_precision": [], "val_recall": []}
    for epoch in range(num_epochs):
        ctime = time.time()
        # Train the model
        # Split the training data into batches
        trained = 0
        total_loss_per_output = np.zeros(out_probas - omit_train_probas, dtype=np.float64)
        true_negative_per_output = np.zeros(out_probas - omit_train_probas, dtype=np.int64)
        false_negative_per_output = np.zeros(out_probas - omit_train_probas, dtype=np.int64)
        true_positive_per_output = np.zeros(out_probas - omit_train_probas, dtype=np.int64)
        false_positive_per_output = np.zeros(out_probas - omit_train_probas, dtype=np.int64)
        total_cumulative_loss = 0.0

        # Shuffle
        training_entries_shuffle = rng.permutation(training_entries)

        steps = 0
        while trained < len(training_entries):
            with torch.no_grad():
                batch_end = min(trained + batch_size, len(training_entries))
                train_image_data_batch = torch.zeros((batch_end - trained, in_channels, image_height, image_width), dtype=torch.float32, device=config.device)
                train_image_ground_truth_batch = torch.zeros((batch_end - trained, image_height, image_width), dtype=torch.float32, device=config.device)
                train_foreground_mask_batch = torch.zeros((batch_end - trained, image_height, image_width), dtype=torch.float32, device=config.device)

                for k in range(trained, batch_end):
                    train_image_data_batch[k - trained, :, :, :] = torch.tensor(dataset_loader.get_image_data(training_entries_shuffle[k]), dtype=torch.float32, device=config.device).permute(2, 0, 1)
                    seg_mask = dataset_loader.get_segmentation_mask(training_entries_shuffle[k], ground_truth)
                    train_image_ground_truth_batch[k - trained, :, :] = torch.tensor(seg_mask, dtype=torch.float32, device=config.device)

                    foreground = dataset_loader.get_segmentation_mask(training_entries_shuffle[k], background)
                    foreground_mask = torch.tensor(foreground, dtype=torch.float32, device=config.device)
                    train_foreground_mask_batch[k - trained, :, :] = foreground_mask

                if rotation_augmentation:
                    # flip the images
                    if np.random.uniform(0, 1) < 0.5:
                        train_image_data_batch = torch.flip(train_image_data_batch, dims=[3])
                        train_image_ground_truth_batch = torch.flip(train_image_ground_truth_batch, dims=[2])
                        train_foreground_mask_batch = torch.flip(train_foreground_mask_batch, dims=[2])

                    # apply elastic deformation
                    train_image_data_batch = torch.nn.functional.pad(train_image_data_batch, (image_height-1, image_height-1, image_width-1, image_width-1), mode="reflect")
                    train_image_ground_truth_batch = torch.nn.functional.pad(train_image_ground_truth_batch, (image_height-1, image_height-1, image_width-1, image_width-1), mode="reflect")
                    train_foreground_mask_batch = torch.nn.functional.pad(train_foreground_mask_batch, (image_height-1, image_height-1, image_width-1, image_width-1), mode="reflect")

                    displacement_field = model_simple_unet.generate_displacement_field()
                    train_image_data_batch = torchvision.transforms.functional.elastic_transform(train_image_data_batch, displacement_field)
                    train_image_ground_truth_batch = torchvision.transforms.functional.elastic_transform(train_image_ground_truth_batch.unsqueeze(1), displacement_field).squeeze(1)
                    train_foreground_mask_batch = torchvision.transforms.functional.elastic_transform(train_foreground_mask_batch.unsqueeze(1), displacement_field).squeeze(1)


                    train_image_data_batch = train_image_data_batch[..., image_height-1:-image_height+1, image_width-1:-image_width+1]
                    train_image_ground_truth_batch = train_image_ground_truth_batch[..., image_height-1:-image_height+1, image_width-1:-image_width+1]
                    train_foreground_mask_batch = train_foreground_mask_batch[..., image_height-1:-image_height+1, image_width-1:-image_width+1]

                    # apply rotation
                    angle_in_deg = np.random.uniform(0, 360)
                    with torch.no_grad():
                        train_image_data_batch = torchvision.transforms.functional.rotate(train_image_data_batch, angle_in_deg)
                        train_image_ground_truth_batch = torchvision.transforms.functional.rotate(train_image_ground_truth_batch, angle_in_deg)
                        train_foreground_mask_batch = torchvision.transforms.functional.rotate(train_foreground_mask_batch, angle_in_deg)

                        rads = np.radians(angle_in_deg % 90.0)
                        lims = 0.5 / (np.sin(rads) + np.cos(rads))
                        # Restrict to (centerx - imagewidth * lims, centery - imageheight * lims) to (centerx + imagewidth * lims, centery + imageheight * lims)
                        ymin = int(image_height // 2 - image_height * lims)
                        ymax = int(image_height // 2 + image_height * lims)
                        xmin = int(image_width // 2 - image_width * lims)
                        xmax = int(image_width // 2 + image_width * lims)

                        xmax = image_pixels_round * ((xmax - xmin) // image_pixels_round) + xmin
                        ymax = image_pixels_round * ((ymax - ymin) // image_pixels_round) + ymin

                        train_image_data_batch = train_image_data_batch[:, :, ymin:ymax, xmin:xmax]
                        train_image_ground_truth_batch = train_image_ground_truth_batch[:, ymin:ymax, xmin:xmax]
                        train_foreground_mask_batch = train_foreground_mask_batch[:, ymin:ymax, xmin:xmax]

                    gc.collect()
                    torch.cuda.empty_cache()

            true_positive, false_positive, false_negative, true_negative, loss, cum_loss = model.training(train_image_data_batch, train_image_ground_truth_batch,
                                                                        foreground_mask=train_foreground_mask_batch, foreground_weight=foreground_weight,
                                                                        background_weight=background_weight, frozen_outputs=omit_train_probas, not_first_epoch=(epoch > 0))

            true_positive_per_output = true_positive + true_positive_per_output
            false_positive_per_output = false_positive + false_positive_per_output
            false_negative_per_output = false_negative + false_negative_per_output
            true_negative_per_output = true_negative + true_negative_per_output

            total_loss_per_output = total_loss_per_output + loss
            total_cumulative_loss += cum_loss

            trained += batch_size

            gc.collect()
            torch.cuda.empty_cache()

        total_loss_per_output /= len(training_entries)
        total_cumulative_loss /= len(training_entries)

        accuracy_per_output = (true_positive_per_output + true_negative_per_output).astype(np.float64) / (true_positive_per_output + true_negative_per_output + false_positive_per_output + false_negative_per_output)
        precision_per_output = true_positive_per_output.astype(np.float64) / (true_positive_per_output + false_positive_per_output)
        recall_per_output = true_positive_per_output.astype(np.float64) / (true_positive_per_output + false_negative_per_output)

        accuracy_per_output = np.nan_to_num(accuracy_per_output, nan=0.0, posinf=0.0, neginf=0.0)
        precision_per_output = np.nan_to_num(precision_per_output, nan=0.0, posinf=0.0, neginf=0.0)
        recall_per_output = np.nan_to_num(recall_per_output, nan=0.0, posinf=0.0, neginf=0.0)

        train_history["loss_cum"].append(total_cumulative_loss)
        train_history["loss"].append(total_loss_per_output)
        train_history["accuracy"].append(accuracy_per_output)
        train_history["precision"].append(precision_per_output)
        train_history["recall"].append(recall_per_output)



        # Test the model
        with torch.no_grad():
            tested = 0
            total_loss_per_output = np.zeros(out_probas - omit_train_probas, dtype=np.float64)
            true_negative_per_output = np.zeros(out_probas - omit_train_probas, dtype=np.int64)
            false_negative_per_output = np.zeros(out_probas - omit_train_probas, dtype=np.int64)
            true_positive_per_output = np.zeros(out_probas - omit_train_probas, dtype=np.int64)
            false_positive_per_output = np.zeros(out_probas - omit_train_probas, dtype=np.int64)
            total_cumulative_loss = 0.0

            while tested < len(validation_entries):
                batch_end = min(tested + batch_size, len(validation_entries))
                test_image_data_batch = torch.zeros((batch_end - tested, in_channels, image_height, image_width), dtype=torch.float32, device=config.device)
                test_image_ground_truth_batch = torch.zeros((batch_end - tested, image_height, image_width), dtype=torch.float32, device=config.device)
                test_foreground_mask_batch = torch.zeros((batch_end - tested, image_height, image_width), dtype=torch.float32, device=config.device)

                for k in range(tested, batch_end):
                    test_image_data_batch[k - tested, :, :, :] = torch.tensor(dataset_loader.get_image_data(validation_entries[k]), dtype=torch.float32, device=config.device).permute(2, 0, 1)
                    seg_mask = dataset_loader.get_segmentation_mask(validation_entries[k], ground_truth)
                    test_image_ground_truth_batch[k - tested, :, :] = torch.tensor(seg_mask, dtype=torch.float32, device=config.device)

                    foreground = dataset_loader.get_segmentation_mask(validation_entries[k], background)
                    foreground_mask = torch.tensor(foreground, dtype=torch.float32, device=config.device)
                    test_foreground_mask_batch[k - tested, :, :] = foreground_mask

                y_pred = model.inference(test_image_data_batch)
                true_positive, false_positive, false_negative, true_negative, loss, cum_loss = model.compute_metrics(y_pred,
                        test_image_ground_truth_batch, foreground_mask, foreground_weight, background_weight, frozen_outputs=omit_train_probas)

                true_positive_per_output = true_positive + true_positive_per_output
                false_positive_per_output = false_positive + false_positive_per_output
                false_negative_per_output = false_negative + false_negative_per_output
                true_negative_per_output = true_negative + true_negative_per_output
                total_loss_per_output = total_loss_per_output + loss
                total_cumulative_loss += cum_loss

                tested += batch_size

                gc.collect()
                torch.cuda.empty_cache()

            total_loss_per_output /= len(validation_entries)
            total_cumulative_loss /= len(validation_entries)

            accuracy_per_output = (true_positive_per_output + true_negative_per_output).astype(np.float64) / (true_positive_per_output + true_negative_per_output + false_positive_per_output + false_negative_per_output)
            precision_per_output = true_positive_per_output.astype(np.float64) / (true_positive_per_output + false_positive_per_output)
            recall_per_output = true_positive_per_output.astype(np.float64) / (true_positive_per_output + false_negative_per_output)

            accuracy_per_output = np.nan_to_num(accuracy_per_output, nan=0.0, posinf=0.0, neginf=0.0)
            precision_per_output = np.nan_to_num(precision_per_output, nan=0.0, posinf=0.0, neginf=0.0)
            recall_per_output = np.nan_to_num(recall_per_output, nan=0.0, posinf=0.0, neginf=0.0)

            train_history["val_loss_cum"].append(total_cumulative_loss)
            train_history["val_loss"].append(total_loss_per_output)
            train_history["val_accuracy"].append(accuracy_per_output)
            train_history["val_precision"].append(precision_per_output)
            train_history["val_recall"].append(recall_per_output)

        print("Time Elapsed: {}".format(time.time() - ctime))
        print("Epoch: {}/{}".format(epoch, num_epochs))
        print("{} (Loss Cum)".format(train_history["loss_cum"][-1]))
        print("{} (Val Loss Cum)".format(train_history["val_loss_cum"][-1]))
        print("{} (Loss)".format(train_history["loss"][-1]))
        print("{} (Val Loss)".format(train_history["val_loss"][-1]))
        print("{} (Accuracy)".format(train_history["accuracy"][-1]))
        print("{} (Val Accuracy)".format(train_history["val_accuracy"][-1]))
        print("{} (Precision)".format(train_history["precision"][-1]))
        print("{} (Val Precision)".format(train_history["val_precision"][-1]))
        print("{} (Recall)".format(train_history["recall"][-1]))
        print("{} (Val Recall)".format(train_history["val_recall"][-1]))
        print("Learning Rate: {}".format(args.learning_rate))
        print("")

        ctime = time.time()

        del train_image_data_batch, train_image_ground_truth_batch, test_image_data_batch, test_image_ground_truth_batch
        gc.collect()
        torch.cuda.empty_cache()

        # Save the model and optimizer
        if epoch % epochs_per_save == 0 and epoch > 0:
            model.save_model(model_dir, epoch)

    print("Training Complete")

    # Save the model and optimizer
    model.save_model(model_dir)
    # Save the training history by converting it to a dataframe
    train_history = convert_train_history(train_history, args.out_probas, args.omit_train_probas)
    train_history = pd.DataFrame(train_history)
    train_history.to_csv(os.path.join(model_dir, "train_history.csv"), index=False)
    # Save the model config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # Plot the training history

    fig, axes = plt.subplots(1 + 2 * (args.out_probas - args.omit_train_probas), 1, figsize=(12, 4 + 8 * (args.out_probas - args.omit_train_probas)))
    # Plot the cum loss
    axes[0].plot(train_history["loss_cum"], label="Train")
    axes[0].plot(train_history["val_loss_cum"], label="Validation")
    axes[0].set_title("Cumulative Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    # Plot the loss in one plot, and the accuracy, precision, and recall in another plot
    for k in range(args.omit_train_probas, args.out_probas):
        axes[1 + 2 * (k - args.omit_train_probas)].plot(train_history["loss_{}".format(k)], label="Train")
        axes[1 + 2 * (k - args.omit_train_probas)].plot(train_history["val_loss_{}".format(k)], label="Validation")
        axes[1 + 2 * (k - args.omit_train_probas)].set_title("Loss (Output {})".format(k))
        axes[1 + 2 * (k - args.omit_train_probas)].set_xlabel("Epoch")
        axes[1 + 2 * (k - args.omit_train_probas)].set_ylabel("Loss")
        axes[1 + 2 * (k - args.omit_train_probas)].legend()
        axes[2 + 2 * (k - args.omit_train_probas)].plot(train_history["accuracy_{}".format(k)], label="Train Accuracy")
        axes[2 + 2 * (k - args.omit_train_probas)].plot(train_history["val_accuracy_{}".format(k)], label="Validation Accuracy")
        axes[2 + 2 * (k - args.omit_train_probas)].plot(train_history["precision_{}".format(k)], label="Train Precision")
        axes[2 + 2 * (k - args.omit_train_probas)].plot(train_history["val_precision_{}".format(k)], label="Validation Precision")
        axes[2 + 2 * (k - args.omit_train_probas)].plot(train_history["recall_{}".format(k)], label="Train Recall")
        axes[2 + 2 * (k - args.omit_train_probas)].plot(train_history["val_recall_{}".format(k)], label="Validation Recall")
        axes[2 + 2 * (k - args.omit_train_probas)].set_title("Accuracy, Precision, and Recall (Output {})".format(k))
        axes[2 + 2 * (k - args.omit_train_probas)].set_xlabel("Epoch")
        axes[2 + 2 * (k - args.omit_train_probas)].set_ylabel("Metric")
        axes[2 + 2 * (k - args.omit_train_probas)].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "train_history.png"))