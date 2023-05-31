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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple U-Net model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--rotation_augmentation", action="store_true", help="Whether to use rotation augmentation. Default False.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use. Default 2.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate to use. Default 1e-5.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps. Default 1. If set to -1, accumulate for the whole dataset.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true", help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")

    image_width = 512
    image_height = 512

    model_data_manager.model_add_argparse_arguments(parser)

    args = parser.parse_args()

    model_dir, dataset_loader, training_entries, validation_entries, prev_model_checkpoint_dir, extra_info = model_data_manager.model_get_argparse_arguments(args)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    training_entries = np.array(training_entries, dtype=object)
    validation_entries = np.array(validation_entries, dtype=object)

    model = model_unet_base.UNetClassifier(hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                           use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height).to(device=config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=10)

    if prev_model_checkpoint_dir is not None:
        model_checkpoint_path = os.path.join(prev_model_checkpoint_dir, "model.pt")
        optimizer_checkpoint_path = os.path.join(prev_model_checkpoint_dir, "optimizer.pt")

        model.load_state_dict(torch.load(model_checkpoint_path))
        optimizer.load_state_dict(torch.load(optimizer_checkpoint_path))

        for g in optimizer.param_groups:
            g['lr'] = args.learning_rate

    # Train the model
    train_history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "precision": [], "val_precision": [], "recall": [], "val_recall": []}

    loss_function = torch.nn.BCELoss(reduction="none")

    batch_size = args.batch_size
    num_epochs = args.epochs
    rotation_augmentation = args.rotation_augmentation
    epochs_per_save = args.epochs_per_save
    gradient_accumulation_steps = args.gradient_accumulation_steps
    image_pixels_round = 2 ** args.pyramid_height

    model_config = {
        "model": "model_simple_unet",
        "epochs": num_epochs,
        "rotation_augmentation": rotation_augmentation,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "epochs_per_save": epochs_per_save,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "use_batch_norm": args.use_batch_norm,
        "use_res_conv": args.use_res_conv,
        "hidden_channels": args.hidden_channels,
        "pyramid_height": args.pyramid_height
    }
    for key, value in extra_info.items():
        model_config[key] = value

    # Compute the number of positive and negative pixels in the training data
    with torch.no_grad():
        num_positive_pixels = 0
        num_negative_pixels = 0

        for k in range(len(training_entries)):
            mask_tensor = torch.tensor(dataset_loader.get_segmentation_masks(training_entries[k])["blood_vessel"], dtype = torch.float32, device = config.device)
            num_positive_pixels = num_positive_pixels + torch.sum(mask_tensor)
            num_negative_pixels = num_negative_pixels + torch.sum(1.0 - mask_tensor)
            del mask_tensor

        print("Number of positive pixels: {}".format(num_positive_pixels))
        print("Number of negative pixels: {}".format(num_negative_pixels))

        # Compute the class weights
        positive_weight = num_negative_pixels / (num_positive_pixels + num_negative_pixels)
        negative_weight = num_positive_pixels / (num_positive_pixels + num_negative_pixels)

        print("Positive weight: {}".format(positive_weight))
        print("Negative weight: {}".format(negative_weight))

    rng = np.random.default_rng()
    for epoch in range(num_epochs):
        ctime = time.time()
        # Train the model
        # Split the training data into batches
        trained = 0
        total_loss = 0.0
        true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0

        # Shuffle
        training_entries_shuffle = rng.permutation(training_entries)

        if gradient_accumulation_steps == -1:
            optimizer.zero_grad()

        steps = 0
        while trained < len(training_entries):
            if gradient_accumulation_steps != -1:
                if steps == 0:
                    optimizer.zero_grad()

            batch_end = min(trained + batch_size, len(training_entries))
            train_image_data_batch = torch.zeros((batch_end - trained, 3, image_height, image_width), dtype=torch.float32, device=config.device)
            train_image_ground_truth_batch = torch.zeros((batch_end - trained, image_height, image_width), dtype=torch.float32, device=config.device)

            for k in range(trained, batch_end):
                train_image_data_batch[k - trained, :, :, :] = torch.tensor(dataset_loader.get_image_data(training_entries_shuffle[k]), dtype=torch.float32, device=config.device).permute(2, 0, 1)
                train_image_ground_truth_batch[k - trained, :, :] = torch.tensor(dataset_loader.get_segmentation_masks(training_entries_shuffle[k])["blood_vessel"], dtype=torch.float32, device=config.device)

            if rotation_augmentation:
                angle_in_deg = np.random.uniform(0, 360)
                with torch.no_grad():
                    train_image_data_batch = torchvision.transforms.functional.rotate(train_image_data_batch, angle_in_deg)
                    train_image_ground_truth_batch = torchvision.transforms.functional.rotate(train_image_ground_truth_batch, angle_in_deg)

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

                gc.collect()
                torch.cuda.empty_cache()


            y_pred = model(train_image_data_batch)
            loss = loss_function(y_pred, train_image_ground_truth_batch)
            # Weighted loss, with precomputed weights
            loss = torch.sum((positive_weight * train_image_ground_truth_batch * loss) + (negative_weight * (1.0 - train_image_ground_truth_batch) * loss))
            loss.backward()
            total_loss += loss.item()

            true_positive += int(torch.sum((y_pred > 0.5) & (train_image_ground_truth_batch == 1)).item())
            true_negative += int(torch.sum((y_pred <= 0.5) & (train_image_ground_truth_batch == 0)).item())
            false_positive += int(torch.sum((y_pred > 0.5) & (train_image_ground_truth_batch == 0)).item())
            false_negative += int(torch.sum((y_pred <= 0.5) & (train_image_ground_truth_batch == 1)).item())

            trained += batch_size

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
            true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0
            while tested < len(validation_entries):
                batch_end = min(tested + batch_size, len(validation_entries))
                test_image_data_batch = torch.zeros((batch_end - tested, 3, image_height, image_width), dtype=torch.float32, device=config.device)
                test_image_ground_truth_batch = torch.zeros((batch_end - tested, image_height, image_width), dtype=torch.float32, device=config.device)

                for k in range(tested, batch_end):
                    test_image_data_batch[k - tested, :, :, :] = torch.tensor(dataset_loader.get_image_data(validation_entries[k]), dtype=torch.float32, device=config.device).permute(2, 0, 1)
                    test_image_ground_truth_batch[k - tested, :, :] = torch.tensor(dataset_loader.get_segmentation_masks(validation_entries[k])["blood_vessel"], dtype=torch.float32, device=config.device)

                y_pred = model(test_image_data_batch)
                loss = loss_function(y_pred, test_image_ground_truth_batch)
                # Weighted loss, with precomputed weights
                loss = torch.sum((positive_weight * test_image_ground_truth_batch * loss) + (negative_weight * (1.0 - test_image_ground_truth_batch) * loss))
                total_loss += loss.item()

                true_positive += int(torch.sum((y_pred > 0.5) & (test_image_ground_truth_batch == 1)).item())
                true_negative += int(torch.sum((y_pred <= 0.5) & (test_image_ground_truth_batch == 0)).item())
                false_positive += int(torch.sum((y_pred > 0.5) & (test_image_ground_truth_batch == 0)).item())
                false_negative += int(torch.sum((y_pred <= 0.5) & (test_image_ground_truth_batch == 1)).item())

                tested += batch_size

            total_loss /= len(validation_entries)
            train_history["val_loss"].append(total_loss)
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
        print("Val Loss: {}".format(train_history["val_loss"][-1]))
        print("Accuracy: {}".format(train_history["accuracy"][-1]))
        print("Val Accuracy: {}".format(train_history["val_accuracy"][-1]))
        print("Precision: {}".format(train_history["precision"][-1]))
        print("Val Precision: {}".format(train_history["val_precision"][-1]))
        print("Recall: {}".format(train_history["recall"][-1]))
        print("Val Recall: {}".format(train_history["val_recall"][-1]))
        print("Learning Rate: {}".format(scheduler.get_lr()))
        print("")
        ctime = time.time()

        del train_image_data_batch, train_image_ground_truth_batch, test_image_data_batch, test_image_ground_truth_batch
        gc.collect()
        torch.cuda.empty_cache()

        # Save the model and optimizer
        if epoch % epochs_per_save == 0:
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
        json.dump(model_config, f)

    # Plot the training history
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(train_history["loss"], label="Loss")
    ax1.plot(train_history["val_loss"], label="Val Loss")
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