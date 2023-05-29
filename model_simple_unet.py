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

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate"),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate"),
            torch.nn.ELU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)

class SimpleUNet(torch.torch.nn.Module):

    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.conv0 = Conv(3, 64)
        self.conv1 = Conv(64, 128)
        self.conv2 = Conv(128, 256)
        self.conv3 = Conv(256, 512)
        self.conv4 = Conv(512, 1024)
        self.conv5 = Conv(1024, 512)
        self.conv6 = Conv(512, 256)
        self.conv7 = Conv(256, 128)
        self.conv8 = Conv(128, 64)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.convT0 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.convT1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.convT2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.convT3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.outconv = torch.nn.Conv2d(64, 1, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # contracting path
        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x = self.conv4(self.maxpool(x3))
        # expanding path
        x = self.conv5(torch.concat([self.convT0(x), x3] , dim=1))
        x = self.conv6(torch.concat([self.convT1(x), x2], dim=1))
        x = self.conv7(torch.concat([self.convT2(x), x1], dim=1))
        x = self.conv8(torch.concat([self.convT3(x), x0], dim=1))
        return torch.squeeze(self.sigmoid(self.outconv(x)), dim=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple U-Net model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--save_dir", type=str, default="simple_unet_model", help="Directory to save the model to. Default simple_unet_model.")
    parser.add_argument("--rotation_augmentation", action="store_true", help="Whether to use rotation augmentation. Default False.")

    args = parser.parse_args()

    model = SimpleUNet().to(device=config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    output_model_path = args.save_dir
    if not os.path.exists(output_model_path):
        os.mkdir(output_model_path)

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
        train_image_ground_truth[i, :, :] = torch.tensor(mask, dtype=torch.float32)

    for i in range(len(test_dataset1_info)):
        image = test_dataset1_info.index[i]
        test_image_data[i, :, :, :] = torch.tensor(cv2.imread(os.path.join(config.input_data_path, "train", "{}.tif".format(image))), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = np.load("segmentation_data/{}/masks.npz".format(image))["blood_vessel"]
        assert mask.dtype == bool
        test_image_ground_truth[i, :, :] = torch.tensor(mask, dtype=torch.float32)

    # Train the model
    train_history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "precision": [], "val_precision": [], "recall": [], "val_recall": []}

    loss_function = torch.nn.BCELoss()

    batch_size = 2
    num_epochs = args.epochs
    rotation_augmentation = args.rotation_augmentation

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
                    train_image_ground_truth_batch = torchvision.transforms.functional.rotate(train_image_ground_truth_batch, angle_in_deg)

                    rads = np.radians(angle_in_deg % 90.0)
                    lims = 0.5 / (np.sin(rads) + np.cos(rads))
                    # Restrict to (centerx - imagewidth * lims, centery - imageheight * lims) to (centerx + imagewidth * lims, centery + imageheight * lims)
                    xmin = int(train_image_data.shape[0] // 2 - train_image_data.shape[0] * lims) + 2
                    xmax = int(train_image_data.shape[0] // 2 + train_image_data.shape[0] * lims) - 2
                    ymin = int(train_image_data.shape[1] // 2 - train_image_data.shape[1] * lims) + 2
                    ymax = int(train_image_data.shape[1] // 2 + train_image_data.shape[1] * lims) - 2

                    train_image_data_batch = train_image_data_batch[:, :, xmin:xmax, ymin:ymax]
                    train_image_ground_truth_batch = train_image_ground_truth_batch[:, xmin:xmax, ymin:ymax]

                gc.collect()
                torch.cuda.empty_cache()


            y_pred = model(train_image_data_batch)
            loss = loss_function(y_pred, train_image_ground_truth_batch)
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

