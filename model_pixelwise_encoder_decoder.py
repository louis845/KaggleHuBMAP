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
    parser = argparse.ArgumentParser(description="Train a pixelwise encoder-decoder model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use. Default 2.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate to use. Default 1e-5.")
    model_data_manager.model_add_argparse_arguments(parser)

    args = parser.parse_args()
    model_name = model_data_manager.model_get_argparse_arguments(args)

    encoder = Encoder().to(device=config.device)
    decoder = Decoder().to(device=config.device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)


    data_information = pd.read_csv(os.path.join(config.input_data_path, "tile_meta.csv"), index_col=0)

    dataset1_info = data_information[data_information["dataset"] == 1]

    # split dataset1 into train and test, 60/40
    train_dataset1_info = dataset1_info.sample(frac=0.6, random_state=0)
    test_dataset1_info = dataset1_info.drop(train_dataset1_info.index)

    train_image_data = torch.zeros((len(train_dataset1_info), 3, 512, 512), dtype=torch.float32, device=config.device)
    test_image_data = torch.zeros((len(test_dataset1_info), 3, 512, 512), dtype=torch.float32, device=config.device)

    train_image_ground_truth = torch.zeros((len(train_dataset1_info), 512, 512), dtype=torch.float32, device=config.device)
    test_image_ground_truth = torch.zeros((len(test_dataset1_info), 512, 512), dtype=torch.float32, device=config.device)

    # Load the images
    for i in range(len(train_dataset1_info)):
        image = train_dataset1_info.index[i]
        train_image_data[i, :, :, :] = torch.tensor(cv2.imread(os.path.join(config.input_data_path, "train", "{}.tif".format(image))), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = np.load("segmentation_data/{}/masks.npz".format(image))["blood_vessel"]
        assert mask.dtype == bool
        train_image_ground_truth[i, :, :] = torch.tensor(mask, dtype=torch.float32, device=config.device)

    for i in range(len(test_dataset1_info)):
        image = test_dataset1_info.index[i]
        test_image_data[i, :, :, :] = torch.tensor(cv2.imread(os.path.join(config.input_data_path, "train", "{}.tif".format(image))), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = np.load("segmentation_data/{}/masks.npz".format(image))["blood_vessel"]
        assert mask.dtype == bool
        test_image_ground_truth[i, :, :] = torch.tensor(mask, dtype=torch.float32, device=config.device)

    # Train the model
    train_history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "precision": [], "val_precision": [], "recall": [], "val_recall": []}

    loss_function = torch.nn.BCELoss(reduction="none")

    batch_size = args.batch_size
    num_epochs = args.epochs
    rotation_augmentation = args.rotation_augmentation

    # Compute the number of positive and negative pixels in the training data
    with torch.no_grad():
        num_positive_pixels = torch.sum(train_image_ground_truth)
        num_negative_pixels = torch.sum(1.0 - train_image_ground_truth)
        print("Number of positive pixels: {}".format(num_positive_pixels))
        print("Number of negative pixels: {}".format(num_negative_pixels))

        # Compute the class weights
        positive_weight = num_negative_pixels / (num_positive_pixels + num_negative_pixels)
        negative_weight = num_positive_pixels / (num_positive_pixels + num_negative_pixels)

        print("Positive weight: {}".format(positive_weight))
        print("Negative weight: {}".format(negative_weight))

    for epoch in range(num_epochs):
        ctime = time.time()
        # Train the model
        # Split the training data into batches
        trained = 0
        optimizer.zero_grad()
        total_loss = 0.0
        true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0
        while trained < len(train_dataset1_info):
            train_image_data_batch = train_image_data[trained:trained+batch_size, :, :, :]
            train_image_ground_truth_batch = train_image_ground_truth[trained:trained+batch_size, :, :]

            if rotation_augmentation:
                angle_in_deg = np.random.uniform(0, 360)
                with torch.no_grad():
                    train_image_data_batch = torchvision.transforms.functional.rotate(train_image_data_batch, angle_in_deg)
                    #train_image_ground_truth_batch = ((torchvision.transforms.functional.rotate(train_image_ground_truth_batch * 255.0, angle_in_deg) / 255.0) > 0.5).type(dtype=torch.float32)
                    train_image_ground_truth_batch = torchvision.transforms.functional.rotate(train_image_ground_truth_batch, angle_in_deg)

                    rads = np.radians(angle_in_deg % 90.0)
                    lims = 0.5 / (np.sin(rads) + np.cos(rads))
                    # Restrict to (centerx - imagewidth * lims, centery - imageheight * lims) to (centerx + imagewidth * lims, centery + imageheight * lims)
                    xmin = int(train_image_data.shape[2] // 2 - train_image_data.shape[2] * lims)
                    xmax = int(train_image_data.shape[2] // 2 + train_image_data.shape[2] * lims)
                    ymin = int(train_image_data.shape[3] // 2 - train_image_data.shape[3] * lims)
                    ymax = int(train_image_data.shape[3] // 2 + train_image_data.shape[3] * lims)

                    xmax = 16 * ((xmax - xmin) // 16) + xmin
                    ymax = 16 * ((ymax - ymin) // 16) + ymin

                    train_image_data_batch = train_image_data_batch[:, :, xmin:xmax, ymin:ymax]
                    train_image_ground_truth_batch = train_image_ground_truth_batch[:, xmin:xmax, ymin:ymax]

                gc.collect()
                torch.cuda.empty_cache()


            y_pred = model(train_image_data_batch)
            loss = loss_function(y_pred, train_image_ground_truth_batch)
            # Weighted loss, with precomputed weights
            loss = torch.mean((positive_weight * train_image_ground_truth_batch * loss) + (negative_weight * (1.0 - train_image_ground_truth_batch) * loss))
            loss.backward()
            total_loss += loss.item()

            true_positive += int(torch.sum((y_pred > 0.5) & (train_image_ground_truth_batch == 1)).item())
            true_negative += int(torch.sum((y_pred <= 0.5) & (train_image_ground_truth_batch == 0)).item())
            false_positive += int(torch.sum((y_pred > 0.5) & (train_image_ground_truth_batch == 0)).item())
            false_negative += int(torch.sum((y_pred <= 0.5) & (train_image_ground_truth_batch == 1)).item())

            trained += batch_size
        optimizer.step()
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
        model.eval()
        with torch.no_grad():
            tested = 0
            total_loss = 0.0
            true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0
            while tested < len(test_dataset1_info):
                test_image_data_batch = test_image_data[tested:tested+batch_size, :, :, :]
                test_image_ground_truth_batch = test_image_ground_truth[tested:tested+batch_size, :, :]

                y_pred = model(test_image_data_batch)
                loss = loss_function(y_pred, test_image_ground_truth_batch)
                # Weighted loss, with precomputed weights
                loss = torch.mean((positive_weight * test_image_ground_truth_batch * loss) + (negative_weight * (1.0 - test_image_ground_truth_batch) * loss))
                total_loss += loss.item()

                true_positive += int(torch.sum((y_pred > 0.5) & (test_image_ground_truth_batch == 1)).item())
                true_negative += int(torch.sum((y_pred <= 0.5) & (test_image_ground_truth_batch == 0)).item())
                false_positive += int(torch.sum((y_pred > 0.5) & (test_image_ground_truth_batch == 0)).item())
                false_negative += int(torch.sum((y_pred <= 0.5) & (test_image_ground_truth_batch == 1)).item())

                tested += batch_size
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
        print("")
        ctime = time.time()

        del train_image_data_batch, train_image_ground_truth_batch, test_image_data_batch, test_image_ground_truth_batch
        gc.collect()
        torch.cuda.empty_cache()

        torch.save(model.state_dict(), os.path.join(output_model_path, "model_epoch{}.pt".format(epoch)))
        torch.save(optimizer.state_dict(), os.path.join(output_model_path, "optimizer_epoch{}.pt".format(epoch)))

    print("Training Complete")

    # Save the model and optimizer
    torch.save(model.state_dict(), os.path.join(output_model_path, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(output_model_path, "optimizer.pt"))
    # Save the training history by converting it to a dataframe
    train_history = pd.DataFrame(train_history)
    train_history.to_csv(os.path.join(output_model_path, "train_history.csv"), index=False)

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