"""A pixelwise encoder-decoder model for semi-supervised image segmentation."""

import gc
import os
import time
import argparse

import config
import model_data_manager

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn
import torchvision
import torchvision.transforms.functional


class Encoder(torch.nn.Module):
    """Fully connected feedforward neural network encoder. 3 -> 16 -> 16 -> 1"""
    def __init__(self, encoder_dim=1, activation=torch.nn.ReLU()):
        super().__init__()

        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, encoder_dim)
        self.activation = activation
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class Decoder(torch.nn.Module):
    """Fully connected feedforward neural network decoder. 15 -> 16 -> 16 -> 3. The first dimension is the image, the remaining are the 1 hot representation of the class."""
    def __init__(self, decoder_dim=1, activation=torch.nn.ReLU()):
        super().__init__()

        self.fc1 = torch.nn.Linear(decoder_dim + 14, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 3)
        self.activation = activation
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pixelwise encoder-decoder model for semi-supervised image segmentation.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use. Default 2.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate to use. Default 1e-5.")

    model_data_manager.model_add_argparse_arguments(parser, allow_missing_validation=True)

    args = parser.parse_args()

    model_dir, dataset_dir, training_entries, validation_entries = model_data_manager.model_get_argparse_arguments(args, allow_missing_validation=True)

    requires_validation = validation_entries is not None

    encoder_model = Encoder().to(device=config.device)
    decoder_model = Decoder().to(device=config.device)

    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=10)

    train_image_data = torch.zeros((len(training_entries), 512, 512, 3), dtype=torch.float32, device=config.device)
    if requires_validation:
        test_image_data = torch.zeros((len(validation_entries), 512, 512, 3), dtype=torch.float32, device=config.device)

    # Load the images and compute the one-hot classes
    for i in range(len(training_entries)):
        image = training_entries[i]
        train_image_data[i, :, :, :] = torch.tensor(cv2.imread(os.path.join(dataset_dir, "{}.tif".format(image))),
                                                    dtype=torch.float32) / 255.0
    training_wsi = torch.tensor(model_data_manager.data_information.loc[training_entries, "source_wsi"], dtype=torch.long, device=config.device) - 1
    training_wsi_one_hot = torch.nn.functional.one_hot(training_wsi, num_classes=14).to(dtype=torch.float32)

    if requires_validation:
        for i in range(len(validation_entries)):
            image = validation_entries[i]
            test_image_data[i, :, :, :] = torch.tensor(cv2.imread(os.path.join(dataset_dir, "{}.tif".format(image))),
                                                       dtype=torch.float32) / 255.0
        validation_wsi = torch.tensor(model_data_manager.data_information.loc[validation_entries, "source_wsi"], dtype=torch.long, device=config.device) - 1
        validation_wsi_one_hot = torch.nn.functional.one_hot(validation_wsi, num_classes=14).to(dtype=torch.float32)

    # Train the model
    train_history = {"loss": [], "val_loss": [], "max_loss": [], "val_max_loss": [], "batch_max_loss": [], "val_batch_max_loss": []}

    batch_size = args.batch_size
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        ctime = time.time()
        # Train the model
        # Split the training data into batches
        trained = 0
        optimizer.zero_grad()
        total_loss = 0.0
        total_max_loss = 0.0
        total_batch_max_loss = 0.0
        while trained < len(training_entries):
            train_image_data_batch = train_image_data[trained:trained + batch_size, :, :, :]

            latent = encoder_model(train_image_data_batch)
            training_wsi_one_hot_batch = training_wsi_one_hot[trained:trained + batch_size, :].unsqueeze(1)\
                .unsqueeze(1).expand(-1, 512, 512, -1)
            reconstruction = decoder_model(torch.concat([latent, training_wsi_one_hot_batch], dim=3))


            # Weighted loss, with precomputed weights
            diff = torch.abs(reconstruction - train_image_data_batch)
            loss = torch.mean(diff ** 2, dim=[1, 2, 3])
            loss = torch.sum(loss)

            loss.backward()
            total_loss += loss.item()
            total_max_loss = max(total_max_loss, torch.max(diff).item())
            total_batch_max_loss += torch.sum(torch.max(diff, dim=[1, 2, 3])).item()

            trained += batch_size

            gc.collect()
            torch.cuda.empty_cache()
        optimizer.step()
        scheduler.step()

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
                    test_image_data_batch = test_image_data[tested:tested + batch_size, :, :, :]

                    latent = encoder_model(test_image_data_batch)
                    validation_wsi_one_hot_batch = validation_wsi_one_hot[tested:tested + batch_size, :].unsqueeze(1)\
                        .unsqueeze(1).expand(-1, 512, 512, -1)
                    reconstruction = decoder_model(torch.concat([latent, validation_wsi_one_hot_batch], dim=3))

                    # Weighted loss, with precomputed weights
                    diff = torch.abs(reconstruction - test_image_data_batch)
                    loss = torch.mean(diff ** 2, dim=[1, 2, 3])
                    loss = torch.sum(loss)

                    total_loss += loss.item()
                    total_max_loss = max(total_max_loss, torch.max(diff).item())
                    total_batch_max_loss += torch.sum(torch.max(diff, dim=[1, 2, 3])).item()

                    tested += batch_size

                    gc.collect()
                    torch.cuda.empty_cache()

            train_history["val_loss"].append(total_loss)
            train_history["val_max_loss"].append(total_max_loss)
            train_history["val_batch_max_loss"].append(total_batch_max_loss)
        else:
            train_history["val_loss"].append(0.0)
            train_history["val_max_loss"].append(0.0)
            train_history["val_batch_max_loss"].append(0.0)


        print("Time Elapsed: {}".format(time.time() - ctime))
        print("Epoch: {}/{}".format(epoch, num_epochs))
        print("Loss: {}".format(train_history["loss"][-1]))
        print("Val Loss: {}".format(train_history["val_loss"][-1]))
        print("Max Loss: {}".format(train_history["max_loss"][-1]))
        print("Val Max Loss: {}".format(train_history["val_max_loss"][-1]))
        print("Batch Max Loss: {}".format(train_history["batch_max_loss"][-1]))
        print("Val Batch Max Loss: {}".format(train_history["val_batch_max_loss"][-1]))
        print("Learning Rate: {}".format(scheduler.get_lr()))
        print("")
        ctime = time.time()

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(encoder_model.state_dict(), os.path.join(model_dir, "encoder_epoch{}.pt".format(epoch)))
        torch.save(decoder_model.state_dict(), os.path.join(model_dir, "decoder_epoch{}.pt".format(epoch)))
        torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_epoch{}.pt".format(epoch)))

    print("Training Complete")

    # Save the model and optimizer
    torch.save(encoder_model.state_dict(), os.path.join(model_dir, "encoder.pt"))
    torch.save(decoder_model.state_dict(), os.path.join(model_dir, "decoder.pt"))
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