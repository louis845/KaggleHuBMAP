# A simple 512 -> 512 U-Net model
import gc
import os
import time
import argparse

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

    image_width = 512
    image_height = 512

    model_data_manager.model_add_argparse_arguments(parser, allow_missing_validation=True)

    args = parser.parse_args()

    model_dir, dataset_loader, training_entries, validation_entries = model_data_manager.model_get_argparse_arguments(args, allow_missing_validation=True)

    requires_validation = validation_entries is not None

    assert type(training_entries) == list
    training_entries = np.array(training_entries, dtype=object)

    if requires_validation:
        assert type(validation_entries) == list
        validation_entries = np.array(validation_entries, dtype=object)

    model_encoder = model_unet_base.UNetEncoder(hidden_channels=64, in_channels=3).to(device=config.device)
    model_decoder = model_unet_base.UNetDecoder(hidden_channels=64, in_channels=3).to(device=config.device)
    optimizer = torch.optim.Adam(list(model_encoder.parameters()) + list(model_decoder.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=10)

    # Train the model
    if requires_validation:
        train_history = {"loss": [], "val_loss": [], "max_loss": [], "val_max_loss": [], "batch_max_loss": [], "val_batch_max_loss": []}
    else:
        train_history = {"loss": [], "max_loss": [], "batch_max_loss": []}

    batch_size = args.batch_size
    num_epochs = args.epochs
    rotation_augmentation = args.rotation_augmentation
    epochs_per_save = args.epochs_per_save
    gradient_accumulation_steps = args.gradient_accumulation_steps

    rng = np.random.default_rng()
    for epoch in range(num_epochs):
        ctime = time.time()
        # Train the model
        # Split the training data into batches
        trained = 0
        total_loss = 0.0
        total_max_loss = 0.0
        total_batch_max_loss = 0.0

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

            for k in range(trained, batch_end):
                train_image_data_batch[k - trained, :, :, :] = torch.tensor(dataset_loader.get_image_data(training_entries_shuffle[k]), dtype=torch.float32, device=config.device).permute(2, 0, 1) / 255.0

            if rotation_augmentation:
                angle_in_deg = np.random.uniform(0, 360)
                with torch.no_grad():
                    train_image_data_batch = torchvision.transforms.functional.rotate(train_image_data_batch, angle_in_deg)

                    rads = np.radians(angle_in_deg % 90.0)
                    lims = 0.5 / (np.sin(rads) + np.cos(rads))
                    # Restrict to (centerx - imagewidth * lims, centery - imageheight * lims) to (centerx + imagewidth * lims, centery + imageheight * lims)
                    ymin = int(image_height // 2 - image_height * lims)
                    ymax = int(image_height // 2 + image_height * lims)
                    xmin = int(image_width // 2 - image_width * lims)
                    xmax = int(image_width // 2 + image_width * lims)

                    xmax = 16 * ((xmax - xmin) // 16) + xmin
                    ymax = 16 * ((ymax - ymin) // 16) + ymin

                    train_image_data_batch = train_image_data_batch[:, :, ymin:ymax, xmin:xmax]

                gc.collect()
                torch.cuda.empty_cache()

            latent_x0, latent_x1, latent_x2, latent_x3, latent_x4 = model_encoder(train_image_data_batch)
            reconstruction = model_decoder(latent_x0, latent_x1, latent_x2, latent_x3, latent_x4)
            diff = torch.abs(train_image_data_batch - reconstruction)

            # Compute losses
            loss = torch.sum(diff)
            loss.backward()
            total_loss += loss.item()
            total_max_loss = max(total_max_loss, torch.max(diff).item())
            total_batch_max_loss += torch.sum(torch.amax(diff, dim=[1, 2, 3])).item()

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
        total_batch_max_loss /= len(training_entries)

        train_history["loss"].append(total_loss)
        train_history["max_loss"].append(total_max_loss)
        train_history["batch_max_loss"].append(total_batch_max_loss)

        # Test the model
        if requires_validation:
            with torch.no_grad():
                tested = 0
                total_loss = 0.0
                total_max_loss = 0.0
                total_batch_max_loss = 0.0
                while tested < len(validation_entries):
                    batch_end = min(tested + batch_size, len(validation_entries))
                    test_image_data_batch = torch.zeros((batch_end - tested, 3, image_height, image_width), dtype=torch.float32, device=config.device)

                    for k in range(tested, batch_end):
                        test_image_data_batch[k - tested, :, :, :] = torch.tensor(dataset_loader.get_image_data(validation_entries[k]), dtype=torch.float32, device=config.device).permute(2, 0, 1) / 255.0

                    latent_x0, latent_x1, latent_x2, latent_x3, latent_x4 = model_encoder(train_image_data_batch)
                    reconstruction = model_decoder(latent_x0, latent_x1, latent_x2, latent_x3, latent_x4)
                    diff = torch.abs(train_image_data_batch - reconstruction)

                    # Compute losses
                    loss = torch.sum(diff)
                    total_loss += loss.item()
                    total_max_loss = max(total_max_loss, torch.max(diff).item())
                    total_batch_max_loss += torch.sum(torch.amax(diff, dim=[1, 2, 3])).item()

                    tested += batch_size

                total_loss /= len(validation_entries)
                total_batch_max_loss /= len(validation_entries)

                train_history["val_loss"].append(total_loss)
                train_history["val_max_loss"].append(total_max_loss)
                train_history["val_batch_max_loss"].append(total_batch_max_loss)

        print("Time Elapsed: {}".format(time.time() - ctime))
        print("Epoch: {}/{}".format(epoch, num_epochs))
        print("Loss: {}".format(train_history["loss"][-1]))
        print("Max Loss: {}".format(train_history["max_loss"][-1]))
        print("Batch Max Loss: {}".format(train_history["batch_max_loss"][-1]))
        if requires_validation:
            print("Val Loss: {}".format(train_history["val_loss"][-1]))
            print("Val Max Loss: {}".format(train_history["val_max_loss"][-1]))
            print("Val Batch Max Loss: {}".format(train_history["val_batch_max_loss"][-1]))
        print("")
        ctime = time.time()

        del train_image_data_batch, test_image_data_batch
        gc.collect()
        torch.cuda.empty_cache()

        # Save the model and optimizer
        if epoch % epochs_per_save == 0:
            torch.save(model_encoder.state_dict(), os.path.join(model_dir, "encoder_epoch{}.pt".format(epoch)))
            torch.save(model_decoder.state_dict(), os.path.join(model_dir, "decoder_epoch{}.pt".format(epoch)))
            torch.save(model_encoder.backbone.state_dict(), os.path.join(model_dir, "backbone_epoch{}.pt".format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_epoch{}.pt".format(epoch)))

    print("Training Complete")

    # Save the model and optimizer
    torch.save(model_encoder.state_dict(), os.path.join(model_dir, "encoder.pt"))
    torch.save(model_decoder.state_dict(), os.path.join(model_dir, "decoder.pt"))
    torch.save(model_encoder.backbone.state_dict(), os.path.join(model_dir, "backbone.pt"))
    torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pt"))
    # Save the training history by converting it to a dataframe
    train_history = pd.DataFrame(train_history)
    train_history.to_csv(os.path.join(model_dir, "train_history.csv"), index=False)

    # Plot the training history
    plt.figure(figsize=(20, 10))
    plt.plot(train_history["loss"], label="Loss")
    plt.plot(train_history["val_loss"], label="Val Loss")
    plt.plot(train_history["accuracy"], label="Accuracy")
    plt.plot(train_history["val_accuracy"], label="Val Accuracy")
    plt.plot(train_history["precision"], label="Precision")
    plt.plot(train_history["val_precision"], label="Val Precision")
    plt.plot(train_history["recall"], label="Recall")
    plt.plot(train_history["val_recall"], label="Val Recall")
    plt.legend()

    plt.show()