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
import image_sampling

def compute_background_foreground(training_entries, dataset_loader, ground_truth_mask_data, foreground_mask_data):
    num_background_positive_pixels = 0
    num_background_negative_pixels = 0
    num_foreground_positive_pixels = 0
    num_foreground_negative_pixels = 0
    num_dset1_entries = 0

    for k in range(len(training_entries)):
        if model_data_manager.data_information.loc[training_entries[k], "dataset"] == 1:
            num_dset1_entries += 1
            seg_mask = dataset_loader.get_segmentation_mask(training_entries[k], ground_truth_mask_data)
            foreground_mask = dataset_loader.get_segmentation_mask(training_entries[k], foreground_mask_data)

            num_foreground_positive_pixels += np.sum(np.logical_and(seg_mask, foreground_mask))
            num_foreground_negative_pixels += np.sum(np.logical_and(np.logical_not(seg_mask), foreground_mask))
            num_background_positive_pixels += np.sum(np.logical_and(seg_mask, np.logical_not(foreground_mask)))
            num_background_negative_pixels += np.sum(
                np.logical_and(np.logical_not(seg_mask), np.logical_not(foreground_mask)))

    return num_background_positive_pixels, num_background_negative_pixels, num_foreground_positive_pixels, num_foreground_negative_pixels, num_dset1_entries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple U-Net model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--rotation_augmentation", action="store_true", help="Whether to use rotation augmentation. Default False.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use. Default 2.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate to use. Default 1e-5.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps. Default 1. If set to -1, accumulate for the whole dataset.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true", help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--use_atrous_conv", action="store_true", help="Whether to use atrous convolutional networks. Default False.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")
    parser.add_argument("--unet_plus", type=str, default="none", help="Whether to use unet plus plus. Available options: none, standard, or deep_supervision. Default none.")
    parser.add_argument("--unet_attention", action="store_true", help="Whether to use attention in the U-Net. Default False. Cannot be used with unet_plus.")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels to use. Default 3.")
    parser.add_argument("--background_weights_split", type=str, help="Whether to use another mask for the background. Default None.", default="None")
    parser.add_argument("--mixup", type=float, default=0.0, help="The alpha value of mixup. Default 0, meaning no mixup.")

    image_width = 512
    image_height = 512

    model_data_manager.model_add_argparse_arguments(parser)

    args = parser.parse_args()

    model_dir, dataset_loader, training_entries, validation_entries, prev_model_checkpoint_dir, extra_info = model_data_manager.model_get_argparse_arguments(args)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    training_entries = np.array(training_entries, dtype=object)
    validation_entries = np.array(validation_entries, dtype=object)

    net_mode = args.unet_plus.lower()
    if net_mode not in ["none", "standard", "deep_supervision"]:
        print("Invalid unet plus mode. The available options are: none, standard, or deep_supervision.")
        exit(1)
    if net_mode != "none" and args.unet_attention:
        print("Cannot use attention with unet plus.")
        exit(1)
    if net_mode != "none" and args.use_atrous_conv:
        print("Cannot use atrous convolution with unet plus.")
        exit(1)

    use_deep_supervision = (net_mode == "deep_supervision")
    if net_mode == "none":
        if args.unet_attention:
            model = model_unet_attention.UNetClassifier(hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                                   use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height, in_channels=args.in_channels, use_atrous_conv=args.use_atrous_conv).to(device=config.device)
        else:
            model = model_unet_base.UNetClassifier(hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                                   use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height, in_channels=args.in_channels, use_atrous_conv=args.use_atrous_conv).to(device=config.device)
    else:
        model = model_unet_plus.UNetClassifier(hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                                  use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                  use_deep_supervision=use_deep_supervision, in_channels=args.in_channels).to(device=config.device)
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.99)
    else:
        print("Invalid optimizer. The available options are: adam, sgd.")
        exit(1)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=10)

    if prev_model_checkpoint_dir is not None:
        model_checkpoint_path = os.path.join(prev_model_checkpoint_dir, "model.pt")
        optimizer_checkpoint_path = os.path.join(prev_model_checkpoint_dir, "optimizer.pt")

        model.load_state_dict(torch.load(model_checkpoint_path))
        optimizer.load_state_dict(torch.load(optimizer_checkpoint_path))

        for g in optimizer.param_groups:
            g['lr'] = args.learning_rate

    # Train the model
    if use_deep_supervision:
        train_history = {"loss": [], "val_loss": []}
        for i in range(args.pyramid_height):
            train_history["accuracy_{}".format(i)] = []
            train_history["val_accuracy_{}".format(i)] = []
            train_history["precision_{}".format(i)] = []
            train_history["val_precision_{}".format(i)] = []
            train_history["recall_{}".format(i)] = []
            train_history["val_recall_{}".format(i)] = []
    else:
        train_history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "precision": [], "val_precision": [], "recall": [], "val_recall": []}

    loss_function = torch.nn.BCELoss(reduction="none")

    batch_size = args.batch_size
    val_batch_size = args.val_batch_size if args.val_batch_size is not None else batch_size
    num_epochs = args.epochs
    rotation_augmentation = args.rotation_augmentation
    epochs_per_save = args.epochs_per_save
    gradient_accumulation_steps = args.gradient_accumulation_steps
    image_pixels_round = 2 ** args.pyramid_height
    in_channels = args.in_channels
    background_weights_split = args.background_weights_split
    mixup = args.mixup

    model_config = {
        "model": "model_simple_unet",
        "epochs": num_epochs,
        "rotation_augmentation": rotation_augmentation,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "epochs_per_save": epochs_per_save,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "use_batch_norm": args.use_batch_norm,
        "use_res_conv": args.use_res_conv,
        "use_atrous_conv": args.use_atrous_conv,
        "hidden_channels": args.hidden_channels,
        "pyramid_height": args.pyramid_height,
        "unet_plus": args.unet_plus,
        "unet_attention": args.unet_attention,
        "in_channels": args.in_channels,
        "background_weights_split": args.background_weights_split,
        "mixup": mixup,
        "training_script": "model_simple_unet.py",
    }
    for key, value in extra_info.items():
        model_config[key] = value

    # Compute the number of positive and negative pixels in the training data
    if background_weights_split == "None":
        background_weights_split = "blood_vessel"
    else:
        if background_weights_split not in dataset_loader.list_segmentation_masks():
            print("Invalid background weights split. The available options are: {}".format(dataset_loader.list_segmentation_masks()))
            exit(1)

    with torch.no_grad():
        num_background_positive_pixels, num_background_negative_pixels, num_foreground_positive_pixels,\
            num_foreground_negative_pixels, num_dset1_entries = compute_background_foreground(training_entries, dataset_loader, "blood_vessel", background_weights_split)

        print("Number of foreground positive pixels: {}".format(num_foreground_positive_pixels))
        print("Number of foreground negative pixels: {}".format(num_foreground_negative_pixels))
        print("Number of background positive pixels: {}".format(num_background_positive_pixels))
        print("Number of background negative pixels: {}".format(num_background_negative_pixels))

        foreground_weight = (num_background_negative_pixels - num_background_positive_pixels) / (num_foreground_positive_pixels - num_foreground_negative_pixels + num_background_negative_pixels - num_background_positive_pixels)
        background_weight = (num_foreground_positive_pixels - num_foreground_negative_pixels) / (num_foreground_positive_pixels - num_foreground_negative_pixels + num_background_negative_pixels - num_background_positive_pixels)

        foreground_weight = 0.5
        background_weight = 0.5

        print("Foreground weight: {}".format(foreground_weight))
        print("Background weight: {}".format(background_weight))

    rng = np.random.default_rng()
    for epoch in range(num_epochs):
        ctime = time.time()
        # Train the model
        # Split the training data into batches
        trained = 0
        total_loss = 0.0
        if use_deep_supervision:
            true_negative_per_level = [0] * args.pyramid_height
            true_positive_per_level = [0] * args.pyramid_height
            false_negative_per_level = [0] * args.pyramid_height
            false_positive_per_level = [0] * args.pyramid_height
        else:
            true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0

        # Shuffle
        training_entries_shuffle = rng.permutation(training_entries)
        if mixup > 0.0:
            training_entries_shuffle2 = rng.permutation(training_entries)

        if gradient_accumulation_steps == -1:
            optimizer.zero_grad()

        steps = 0
        while trained < len(training_entries):
            if gradient_accumulation_steps != -1:
                if steps == 0:
                    optimizer.zero_grad()

            batch_end = min(trained + batch_size, len(training_entries))
            batch_indices = training_entries_shuffle[trained:batch_end]

            if mixup > 0.0:
                batch_indices2 = training_entries_shuffle2[trained:batch_end]
                train_image_data_batch, train_image_ground_truth_batch, train_image_multiclass_gt_batch, train_image_ground_truth_ds_batch, \
                    train_image_multiclass_gt_ds_batch = image_sampling.sample_images_mixup(batch_indices, batch_indices2, dataset_loader, mixup_alpha=mixup,
                                                                                      rotation_augmentation=rotation_augmentation,
                                                                                      multiclass_labels_dict=None,
                                                                                      deep_supervision_downsamples=0,
                                                                                      crop_height=512, crop_width=512)
            else:
                train_image_data_batch, train_image_ground_truth_batch, train_image_multiclass_gt_batch, train_image_ground_truth_ds_batch, \
                    train_image_multiclass_gt_ds_batch = image_sampling.sample_images(batch_indices, dataset_loader,
                                                                                      rotation_augmentation=rotation_augmentation,
                                                                                      multiclass_labels_dict=None,
                                                                                      deep_supervision_downsamples=0,
                                                                                      crop_height=512, crop_width=512)

            gc.collect()
            torch.cuda.empty_cache()


            y_pred = model(train_image_data_batch)

            if use_deep_supervision:
                loss = 0.0
                for k in range(args.pyramid_height):
                    y_pred_level = y_pred[k]
                    loss_level = loss_function(y_pred_level, train_image_ground_truth_batch)
                    # Weighted loss, with precomputed weights
                    loss = torch.sum(loss_level) + loss

                    true_positive_per_level[k] += int(torch.sum((y_pred_level > 0.5) & (train_image_ground_truth_batch == 1)).item())
                    true_negative_per_level[k] += int(torch.sum((y_pred_level <= 0.5) & (train_image_ground_truth_batch == 0)).item())
                    false_positive_per_level[k] += int(torch.sum((y_pred_level > 0.5) & (train_image_ground_truth_batch == 0)).item())
                    false_negative_per_level[k] += int(torch.sum((y_pred_level <= 0.5) & (train_image_ground_truth_batch == 1)).item())

                total_loss += loss.item()
                loss.backward()
            else:
                loss = loss_function(y_pred, train_image_ground_truth_batch)
                # Weighted loss, with precomputed weights
                loss = torch.sum(loss)
                loss.backward()
                total_loss += loss.item()

                true_positive += int(torch.sum((y_pred > 0.5) & (train_image_ground_truth_batch == 1)).item())
                true_negative += int(torch.sum((y_pred <= 0.5) & (train_image_ground_truth_batch == 0)).item())
                false_positive += int(torch.sum((y_pred > 0.5) & (train_image_ground_truth_batch == 0)).item())
                false_negative += int(torch.sum((y_pred <= 0.5) & (train_image_ground_truth_batch == 1)).item())

            trained += len(batch_indices)

            if gradient_accumulation_steps != -1:
                steps += 1
                if steps == gradient_accumulation_steps:
                    optimizer.step()
                    steps = 0
        if gradient_accumulation_steps == -1:
            optimizer.step()
        else:
            if steps > 0:
                optimizer.step()
        scheduler.step()

        total_loss /= len(training_entries)
        train_history["loss"].append(total_loss)

        if use_deep_supervision:
            for k in range(args.pyramid_height):
                train_history["accuracy_" + str(k)].append((true_positive_per_level[k] + true_negative_per_level[k])
                                                           / (true_positive_per_level[k] + true_negative_per_level[k] + false_positive_per_level[k] + false_negative_per_level[k]))
                if true_positive_per_level[k] + false_positive_per_level[k] == 0:
                    train_history["precision_" + str(k)].append(0.0)
                else:
                    train_history["precision_" + str(k)].append(true_positive_per_level[k] / (true_positive_per_level[k] + false_positive_per_level[k]))
                if true_positive_per_level[k] + false_negative_per_level[k] == 0:
                    train_history["recall_" + str(k)].append(0.0)
                else:
                    train_history["recall_" + str(k)].append(true_positive_per_level[k] / (true_positive_per_level[k] + false_negative_per_level[k]))
        else:
            train_history["accuracy"].append((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative))
            if true_positive + false_positive == 0:
                train_history["precision"].append(0.0)
            else:
                train_history["precision"].append(true_positive / (true_positive + false_positive))
            if true_positive + false_negative == 0:
                train_history["recall"].append(0.0)
            else:
                train_history["recall"].append(true_positive / (true_positive + false_negative))

        # Test the model
        with torch.no_grad():
            tested = 0
            total_loss = 0.0
            if use_deep_supervision:
                true_positive_per_level = [0] * args.pyramid_height
                true_negative_per_level = [0] * args.pyramid_height
                false_positive_per_level = [0] * args.pyramid_height
                false_negative_per_level = [0] * args.pyramid_height
            else:
                true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0
            while tested < len(validation_entries):
                batch_end = min(tested + val_batch_size, len(validation_entries))
                batch_indices = validation_entries[tested:batch_end]

                test_image_data_batch, test_image_ground_truth_batch, test_image_multiclass_gt_batch, test_image_ground_truth_ds_batch, \
                    test_image_multiclass_gt_ds_batch = image_sampling.sample_images(batch_indices, dataset_loader,
                                                                                     rotation_augmentation=False,
                                                                                     multiclass_labels_dict=None,
                                                                                     deep_supervision_downsamples=0)

                y_pred = model(test_image_data_batch)

                if use_deep_supervision:
                    for k in range(args.pyramid_height):
                        # Weighted loss, with precomputed weights
                        loss = loss_function(y_pred[k], test_image_ground_truth_batch)
                        loss = torch.sum(loss)
                        total_loss += loss.item()

                        true_positive_per_level[k] += int(torch.sum((y_pred[k] > 0.5) & (test_image_ground_truth_batch == 1)).item())
                        true_negative_per_level[k] += int(torch.sum((y_pred[k] <= 0.5) & (test_image_ground_truth_batch == 0)).item())
                        false_positive_per_level[k] += int(torch.sum((y_pred[k] > 0.5) & (test_image_ground_truth_batch == 0)).item())
                        false_negative_per_level[k] += int(torch.sum((y_pred[k] <= 0.5) & (test_image_ground_truth_batch == 1)).item())
                else:
                    loss = loss_function(y_pred, test_image_ground_truth_batch)
                    # Weighted loss, with precomputed weights
                    loss = torch.sum(loss)
                    total_loss += loss.item()

                    true_positive += int(torch.sum((y_pred > 0.5) & (test_image_ground_truth_batch == 1)).item())
                    true_negative += int(torch.sum((y_pred <= 0.5) & (test_image_ground_truth_batch == 0)).item())
                    false_positive += int(torch.sum((y_pred > 0.5) & (test_image_ground_truth_batch == 0)).item())
                    false_negative += int(torch.sum((y_pred <= 0.5) & (test_image_ground_truth_batch == 1)).item())

                tested += len(batch_indices)

            total_loss /= len(validation_entries)
            train_history["val_loss"].append(total_loss)
            if use_deep_supervision:
                for k in range(args.pyramid_height):
                    train_history["val_accuracy_" + str(k)].append((true_positive_per_level[k] + true_negative_per_level[k]) / (true_positive_per_level[k] + true_negative_per_level[k] + false_positive_per_level[k] + false_negative_per_level[k]))
                    if true_positive_per_level[k] + false_positive_per_level[k] == 0:
                        train_history["val_precision_" + str(k)].append(0.0)
                    else:
                        train_history["val_precision_" + str(k)].append(true_positive_per_level[k] / (true_positive_per_level[k] + false_positive_per_level[k]))
                    if true_positive_per_level[k] + false_negative_per_level[k] == 0:
                        train_history["val_recall_" + str(k)].append(0.0)
                    else:
                        train_history["val_recall_" + str(k)].append(true_positive_per_level[k] / (true_positive_per_level[k] + false_negative_per_level[k]))
            else:
                train_history["val_accuracy"].append((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative))
                if true_positive + false_positive == 0:
                    train_history["val_precision"].append(0.0)
                else:
                    train_history["val_precision"].append(true_positive / (true_positive + false_positive))
                if true_positive + false_negative == 0:
                    train_history["val_recall"].append(0.0)
                else:
                    train_history["val_recall"].append(true_positive / (true_positive + false_negative))

        print("Time Elapsed: {}".format(time.time() - ctime))
        print("Epoch: {}/{}".format(epoch, num_epochs))
        print("Loss: {}".format(train_history["loss"][-1]))
        print("Loss Dset1: {}".format(train_history["loss_dset1"][-1]))
        print("Val Loss: {}".format(train_history["val_loss"][-1]))
        if use_deep_supervision:
            k = args.pyramid_height - 1
            print("Accuracy {}: {}".format(k, train_history["accuracy_" + str(k)][-1]))
            print("Val Accuracy {}: {}".format(k, train_history["val_accuracy_" + str(k)][-1]))
            print("Precision {}: {}".format(k, train_history["precision_" + str(k)][-1]))
            print("Val Precision {}: {}".format(k, train_history["val_precision_" + str(k)][-1]))
            print("Recall {}: {}".format(k, train_history["recall_" + str(k)][-1]))
            print("Val Recall {}: {}".format(k, train_history["val_recall_" + str(k)][-1]))
        else:
            print("Accuracy: {}".format(train_history["accuracy"][-1]))
            print("Val Accuracy: {}".format(train_history["val_accuracy"][-1]))
            print("Precision: {}".format(train_history["precision"][-1]))
            print("Val Precision: {}".format(train_history["val_precision"][-1]))
            print("Recall: {}".format(train_history["recall"][-1]))
            print("Val Recall: {}".format(train_history["val_recall"][-1]))
        print("Learning Rate: {}".format(scheduler.get_lr()))
        print("")

        train_history_save = pd.DataFrame(train_history)
        train_history_save.to_csv(os.path.join(model_dir, "train_history.csv"), index=False)

        ctime = time.time()

        del train_image_data_batch, train_image_ground_truth_batch, test_image_data_batch, test_image_ground_truth_batch
        gc.collect()
        torch.cuda.empty_cache()

        # Save the model and optimizer
        if epoch % epochs_per_save == 0 and epoch > 0:
            torch.save(model.state_dict(), os.path.join(model_dir, "model_epoch{}.pt".format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_epoch{}.pt".format(epoch)))

    print("Training Complete")

    # Save the model and optimizer
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pt"))
    # Save the training history by converting it to a dataframe
    train_history = pd.DataFrame(train_history)
    train_history.to_csv(os.path.join(model_dir, "train_history.csv"), index=False)
    # Save the model config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # Plot the training history. If we use deep supervision we create args.pyramid_height + 1 number of plots.
    if use_deep_supervision:
        fig, axes = plt.subplots(args.pyramid_height + 1, 1, figsize=(12, 4 * (args.pyramid_height + 1)))
        axes[0].plot(train_history["loss"], label="Loss")
        axes[0].plot(train_history["val_loss"], label="Val Loss")
        axes[0].plot(train_history["loss_dset1"], label="Loss Dset1")
        for k in range(args.pyramid_height):
            axes[k+1].plot(train_history["accuracy_" + str(k)], label="Accuracy")
            axes[k+1].plot(train_history["val_accuracy_" + str(k)], label="Val Accuracy")
            axes[k+1].plot(train_history["precision_" + str(k)], label="Precision")
            axes[k+1].plot(train_history["val_precision_" + str(k)], label="Val Precision")
            axes[k+1].plot(train_history["recall_" + str(k)], label="Recall")
            axes[k+1].plot(train_history["val_recall_" + str(k)], label="Val Recall")

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")

        for k in range(args.pyramid_height):
            axes[k+1].set_xlabel("Epoch")
            axes[k+1].set_ylabel("Metric")
            axes[k+1].set_title("Level {}".format(k))
            axes[k+1].legend()

        axes[0].legend()

        plt.savefig(os.path.join(model_dir, "train_history.png"))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(train_history["loss"], label="Loss")
        ax1.plot(train_history["val_loss"], label="Val Loss")
        ax1.plot(train_history["loss_dset1"], label="Loss Dset1")
        ax2.plot(train_history["accuracy"], label="Accuracy")
        ax2.plot(train_history["val_accuracy"], label="Val Accuracy")
        ax2.plot(train_history["precision"], label="Precision")
        ax2.plot(train_history["val_precision"], label="Val Precision")
        ax2.plot(train_history["recall"], label="Recall")
        ax2.plot(train_history["val_recall"], label="Val Recall")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metric")
        ax1.legend()
        ax2.legend()

        plt.savefig(os.path.join(model_dir, "train_history.png"))