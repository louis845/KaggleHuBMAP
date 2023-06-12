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
import model_unet_base
import model_unet_attention
import model_simple_unet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple U-Net model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--rotation_augmentation", action="store_true", help="Whether to use rotation augmentation. Default False.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use. Default 2.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate to use. Default 1e-5.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true", help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--use_atrous_conv", action="store_true", help="Whether to use atrous convolutional networks. Default False.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")
    parser.add_argument("--unet_attention", action="store_true", help="Whether to use attention in the U-Net. Default False. Cannot be used with unet_plus.")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels to use. Default 3.")

    image_width = 512
    image_height = 512

    model_data_manager.model_add_argparse_arguments(parser)

    args = parser.parse_args()

    model_dir, dataset_loader, training_entries, validation_entries, prev_model_checkpoint_dir, extra_info = model_data_manager.model_get_argparse_arguments(args)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    training_entries = np.array(training_entries, dtype=object)
    validation_entries = np.array(validation_entries, dtype=object)

    if args.unet_attention:
        model = model_unet_attention.UNetClassifier(hidden_channels=args.hidden_channels,
                                                    use_batch_norm=args.use_batch_norm,
                                                    use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                    in_channels=args.in_channels,
                                                    use_atrous_conv=args.use_atrous_conv, deep_supervision=True).to(device=config.device)
    else:
        model = model_unet_base.UNetClassifier(hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                               use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                               in_channels=args.in_channels, use_atrous_conv=args.use_atrous_conv, deep_supervision=True).to(device=config.device)

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.99)
    else:
        print("Invalid optimizer. The available options are: adam, sgd.")
        exit(1)

    if prev_model_checkpoint_dir is not None:
        model_checkpoint_path = os.path.join(prev_model_checkpoint_dir, "model.pt")
        optimizer_checkpoint_path = os.path.join(prev_model_checkpoint_dir, "optimizer.pt")

        model.load_state_dict(torch.load(model_checkpoint_path))
        optimizer.load_state_dict(torch.load(optimizer_checkpoint_path))

        for g in optimizer.param_groups:
            g['lr'] = args.learning_rate

    batch_size = args.batch_size
    num_epochs = args.epochs
    rotation_augmentation = args.rotation_augmentation
    epochs_per_save = args.epochs_per_save
    image_pixels_round = 2 ** args.pyramid_height
    in_channels = args.in_channels
    pyr_height = args.pyramid_height

    model_config = {
        "model": "model_progressive_supervised_unet",
        "epochs": num_epochs,
        "rotation_augmentation": rotation_augmentation,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "epochs_per_save": epochs_per_save,
        "use_batch_norm": args.use_batch_norm,
        "use_res_conv": args.use_res_conv,
        "use_atrous_conv": args.use_atrous_conv,
        "hidden_channels": args.hidden_channels,
        "pyramid_height": args.pyramid_height,
        "unet_attention": args.unet_attention,
        "in_channels": args.in_channels,
        "training_script": "model_progressive_supervised_unet.py",
    }
    for key, value in extra_info.items():
        model_config[key] = value

    ground_truth = "blood_vessel"
    background = "blood_vessel"
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

        foreground_weight = 0.5
        background_weight = 0.5

    rng = np.random.default_rng()

    train_history = {"loss_cum": [], "val_loss_cum": []}

    for k in range(pyr_height):
        train_history["loss{}".format(k)] = []
        train_history["accuracy{}".format(k)] = []
        train_history["precision{}".format(k)] = []
        train_history["recall{}".format(k)] = []
        train_history["val_loss{}".format(k)] = []
        train_history["val_accuracy{}".format(k)] = []
        train_history["val_precision{}".format(k)] = []
        train_history["val_recall{}".format(k)] = []

    for epoch in range(num_epochs):
        ctime = time.time()
        # Train the model
        # Split the training data into batches
        trained = 0
        total_cum_loss = 0.0
        total_loss_per_output = np.zeros(pyr_height, dtype=np.float64)
        true_negative_per_output = np.zeros(pyr_height, dtype=np.int64)
        false_negative_per_output = np.zeros(pyr_height, dtype=np.int64)
        true_positive_per_output = np.zeros(pyr_height, dtype=np.int64)
        false_positive_per_output = np.zeros(pyr_height, dtype=np.int64)

        # Shuffle
        training_entries_shuffle = rng.permutation(training_entries)

        steps = 0
        while trained < len(training_entries):
            with torch.no_grad():
                batch_end = min(trained + batch_size, len(training_entries))
                current_batch_size = batch_end - trained
                train_image_data_batch = torch.zeros((current_batch_size, in_channels, image_height, image_width), dtype=torch.float32, device=config.device)
                train_image_ground_truth_batch = torch.zeros((current_batch_size, image_height, image_width), dtype=torch.float32, device=config.device)
                #train_foreground_mask_batch = torch.zeros((batch_end - trained, image_height, image_width), dtype=torch.float32, device=config.device)

                for k in range(trained, batch_end):
                    train_image_data_batch[k - trained, :, :, :] = torch.tensor(dataset_loader.get_image_data(training_entries_shuffle[k]), dtype=torch.float32, device=config.device).permute(2, 0, 1)
                    seg_mask = dataset_loader.get_segmentation_mask(training_entries_shuffle[k], ground_truth)
                    train_image_ground_truth_batch[k - trained, :, :] = torch.tensor(seg_mask, dtype=torch.float32, device=config.device)

                    #foreground = dataset_loader.get_segmentation_mask(training_entries_shuffle[k], background)
                    #foreground_mask = torch.tensor(foreground, dtype=torch.float32, device=config.device)
                    #train_foreground_mask_batch[k - trained, :, :] = foreground_mask

                if rotation_augmentation:
                    # flip the images
                    if np.random.uniform(0, 1) < 0.5:
                        train_image_data_batch = torch.flip(train_image_data_batch, dims=[3])
                        train_image_ground_truth_batch = torch.flip(train_image_ground_truth_batch, dims=[2])
                        #train_foreground_mask_batch = torch.flip(train_foreground_mask_batch, dims=[2])

                    # apply elastic deformation
                    train_image_data_batch = torch.nn.functional.pad(train_image_data_batch, (image_height-1, image_height-1, image_width-1, image_width-1), mode="reflect")
                    train_image_ground_truth_batch = torch.nn.functional.pad(train_image_ground_truth_batch, (image_height-1, image_height-1, image_width-1, image_width-1), mode="reflect")
                    #train_foreground_mask_batch = torch.nn.functional.pad(train_foreground_mask_batch, (image_height-1, image_height-1, image_width-1, image_width-1), mode="reflect")

                    displacement_field = model_simple_unet.generate_displacement_field()
                    train_image_data_batch = torchvision.transforms.functional.elastic_transform(train_image_data_batch, displacement_field)
                    train_image_ground_truth_batch = torchvision.transforms.functional.elastic_transform(train_image_ground_truth_batch.unsqueeze(1), displacement_field).squeeze(1)
                    #train_foreground_mask_batch = torchvision.transforms.functional.elastic_transform(train_foreground_mask_batch.unsqueeze(1), displacement_field).squeeze(1)

                    # apply rotation
                    angle_in_deg = np.random.uniform(0, 360)
                    train_image_data_batch = torchvision.transforms.functional.rotate(train_image_data_batch, angle_in_deg)
                    train_image_ground_truth_batch = torchvision.transforms.functional.rotate(train_image_ground_truth_batch, angle_in_deg)

                    train_image_data_batch = train_image_data_batch[..., image_height - 1:-image_height + 1, image_width - 1:-image_width + 1]
                    train_image_ground_truth_batch = train_image_ground_truth_batch[..., image_height - 1:-image_height + 1, image_width - 1:-image_width + 1]

                # Crop image to random 448x448
                crop_height = 448
                crop_width = 448

                crop_y = np.random.randint(0, image_height - crop_height + 1)
                crop_x = np.random.randint(0, image_width - crop_width + 1)

                train_image_data_batch = train_image_data_batch[..., crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
                train_image_ground_truth_batch = train_image_ground_truth_batch[..., crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

                gc.collect()
                torch.cuda.empty_cache()

            optimizer.zero_grad()
            result, deep_outputs = model(train_image_data_batch)

            crop_height = train_image_ground_truth_batch.shape[1]
            crop_width = train_image_ground_truth_batch.shape[2]
            train_image_ground_truth_pooled_batch = train_image_ground_truth_batch.view(current_batch_size, 1, crop_height, crop_width)
            loss = 0.0
            for k in range(pyr_height - 2, -1, -1):
                with torch.no_grad():
                    train_image_ground_truth_pooled_batch = torch.nn.functional.max_pool2d(train_image_ground_truth_pooled_batch, kernel_size=2, stride=2)

                scale_factor = 2 ** (pyr_height - 1 - k)
                k_loss = torch.nn.functional.binary_cross_entropy(deep_outputs[k], train_image_ground_truth_pooled_batch.view(current_batch_size, crop_height // scale_factor, crop_width // scale_factor),
                                                         reduction="sum") * scale_factor
                loss += k_loss

                total_loss_per_output[k] += k_loss.item()

                with torch.no_grad():
                    true_negative_per_output[k] += ((deep_outputs[k] < 0.5) & (train_image_ground_truth_pooled_batch < 0.5)).sum().item()
                    false_negative_per_output[k] += ((deep_outputs[k] < 0.5) & (train_image_ground_truth_pooled_batch >= 0.5)).sum().item()
                    true_positive_per_output[k] += ((deep_outputs[k] >= 0.5) & (train_image_ground_truth_pooled_batch >= 0.5)).sum().item()
                    false_positive_per_output[k] += ((deep_outputs[k] >= 0.5) & (train_image_ground_truth_pooled_batch < 0.5)).sum().item()



            result_loss = torch.nn.functional.binary_cross_entropy(result, train_image_ground_truth_batch, reduction="sum")
            loss += result_loss

            total_loss_per_output[-1] += result_loss.item()
            with torch.no_grad():
                true_negative_per_output[-1] += ((result < 0.5) & (train_image_ground_truth_batch < 0.5)).sum().item()
                false_negative_per_output[-1] += ((result < 0.5) & (train_image_ground_truth_batch >= 0.5)).sum().item()
                true_positive_per_output[-1] += ((result >= 0.5) & (train_image_ground_truth_batch >= 0.5)).sum().item()
                false_positive_per_output[-1] += ((result >= 0.5) & (train_image_ground_truth_batch < 0.5)).sum().item()

            loss.backward()
            optimizer.step()

            total_cum_loss += loss.item()

            trained += current_batch_size

            gc.collect()
            torch.cuda.empty_cache()

        total_loss_per_output /= len(training_entries)
        total_cum_loss /= len(training_entries)

        accuracy_per_output = (true_positive_per_output + true_negative_per_output).astype(np.float64) / (true_positive_per_output + true_negative_per_output + false_positive_per_output + false_negative_per_output)
        precision_per_output = true_positive_per_output.astype(np.float64) / (true_positive_per_output + false_positive_per_output)
        recall_per_output = true_positive_per_output.astype(np.float64) / (true_positive_per_output + false_negative_per_output)

        accuracy_per_output = np.nan_to_num(accuracy_per_output, nan=0.0, posinf=0.0, neginf=0.0)
        precision_per_output = np.nan_to_num(precision_per_output, nan=0.0, posinf=0.0, neginf=0.0)
        recall_per_output = np.nan_to_num(recall_per_output, nan=0.0, posinf=0.0, neginf=0.0)

        train_history["loss_cum"].append(total_cum_loss)
        for k in range(pyr_height):
            train_history["loss{}".format(k)].append(total_loss_per_output[k])
            train_history["accuracy{}".format(k)].append(accuracy_per_output[k])
            train_history["precision{}".format(k)].append(precision_per_output[k])
            train_history["recall{}".format(k)].append(recall_per_output[k])



        # Test the model
        with torch.no_grad():
            tested = 0
            total_cum_loss = 0.0
            total_loss_per_output = np.zeros(pyr_height, dtype=np.float64)
            true_negative_per_output = np.zeros(pyr_height, dtype=np.int64)
            false_negative_per_output = np.zeros(pyr_height, dtype=np.int64)
            true_positive_per_output = np.zeros(pyr_height, dtype=np.int64)
            false_positive_per_output = np.zeros(pyr_height, dtype=np.int64)

            while tested < len(validation_entries):
                batch_end = min(tested + batch_size, len(validation_entries))
                current_batch_size = batch_end - tested
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

                result, deep_outputs = model(test_image_data_batch)

                crop_height = test_image_ground_truth_batch.shape[1]
                crop_width = test_image_ground_truth_batch.shape[2]
                test_image_ground_truth_pooled_batch = test_image_ground_truth_batch.view(current_batch_size, 1, crop_height,
                                                                                            crop_width)
                loss = 0.0
                for k in range(pyr_height - 2, -1, -1):
                    test_image_ground_truth_pooled_batch = torch.nn.functional.max_pool2d(test_image_ground_truth_pooled_batch, kernel_size=2, stride=2)

                    scale_factor = 2 ** (pyr_height - 1 - k)
                    k_loss = torch.nn.functional.binary_cross_entropy(deep_outputs[k],
                                                                      test_image_ground_truth_pooled_batch.view(
                                                                          current_batch_size, crop_height // scale_factor,
                                                                          crop_width // scale_factor),
                                                                      reduction="sum") * scale_factor
                    loss += k_loss

                    total_loss_per_output[k] += k_loss.item()

                    true_negative_per_output[k] += ((deep_outputs[k] < 0.5) & (
                                test_image_ground_truth_pooled_batch < 0.5)).sum().item()
                    false_negative_per_output[k] += ((deep_outputs[k] < 0.5) & (
                                test_image_ground_truth_pooled_batch >= 0.5)).sum().item()
                    true_positive_per_output[k] += ((deep_outputs[k] >= 0.5) & (
                                test_image_ground_truth_pooled_batch >= 0.5)).sum().item()
                    false_positive_per_output[k] += ((deep_outputs[k] >= 0.5) & (
                                test_image_ground_truth_pooled_batch < 0.5)).sum().item()

                result_loss = torch.nn.functional.binary_cross_entropy(result, test_image_ground_truth_batch,
                                                                       reduction="sum")
                loss += result_loss

                total_loss_per_output[-1] += result_loss.item()
                true_negative_per_output[-1] += ((result < 0.5) & (test_image_ground_truth_batch < 0.5)).sum().item()
                false_negative_per_output[-1] += ((result < 0.5) & (test_image_ground_truth_batch >= 0.5)).sum().item()
                true_positive_per_output[-1] += ((result >= 0.5) & (test_image_ground_truth_batch >= 0.5)).sum().item()
                false_positive_per_output[-1] += ((result >= 0.5) & (test_image_ground_truth_batch < 0.5)).sum().item()

                total_cum_loss += loss.item()

                tested += current_batch_size

                gc.collect()
                torch.cuda.empty_cache()

            total_loss_per_output /= len(validation_entries)
            total_cum_loss /= len(validation_entries)

            accuracy_per_output = (true_positive_per_output + true_negative_per_output).astype(np.float64) / (true_positive_per_output + true_negative_per_output + false_positive_per_output + false_negative_per_output)
            precision_per_output = true_positive_per_output.astype(np.float64) / (true_positive_per_output + false_positive_per_output)
            recall_per_output = true_positive_per_output.astype(np.float64) / (true_positive_per_output + false_negative_per_output)

            accuracy_per_output = np.nan_to_num(accuracy_per_output, nan=0.0, posinf=0.0, neginf=0.0)
            precision_per_output = np.nan_to_num(precision_per_output, nan=0.0, posinf=0.0, neginf=0.0)
            recall_per_output = np.nan_to_num(recall_per_output, nan=0.0, posinf=0.0, neginf=0.0)

            train_history["val_loss_cum"].append(total_cum_loss)
            for k in range(pyr_height):
                train_history["val_loss{}".format(k)].append(total_loss_per_output[k])
                train_history["val_accuracy{}".format(k)].append(accuracy_per_output[k])
                train_history["val_precision{}".format(k)].append(precision_per_output[k])
                train_history["val_recall{}".format(k)].append(recall_per_output[k])

        print("Time Elapsed: {}".format(time.time() - ctime))
        print("Epoch: {}/{}".format(epoch, num_epochs))
        print("{} (Loss Cum)".format(train_history["loss_cum"][-1]))
        print("{} (Val Loss Cum)".format(train_history["val_loss_cum"][-1]))
        print("{} (Result Loss)".format(train_history["loss{}".format(pyr_height - 1)][-1]))
        print("{} (Val Result Loss)".format(train_history["val_loss{}".format(pyr_height - 1)][-1]))
        print("{} (Result Accuracy)".format(train_history["accuracy{}".format(pyr_height - 1)][-1]))
        print("{} (Val Result Accuracy)".format(train_history["val_accuracy{}".format(pyr_height - 1)][-1]))
        print("{} (Result Precision)".format(train_history["precision{}".format(pyr_height - 1)][-1]))
        print("{} (Val Result Precision)".format(train_history["val_precision{}".format(pyr_height - 1)][-1]))
        print("{} (Result Recall)".format(train_history["recall{}".format(pyr_height - 1)][-1]))
        print("{} (Val Result Recall)".format(train_history["val_recall{}".format(pyr_height - 1)][-1]))
        print("Learning Rate: {}".format(args.learning_rate))
        print("")

        # Save the training history
        train_history_save = pd.DataFrame(train_history)
        train_history_save.to_csv(os.path.join(model_dir, "train_history.csv"), index=False)
        del train_history_save

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

    # Plot the training history

    fig, axes = plt.subplots(1 + 2 * (pyr_height), 1, figsize=(12, 4 + 8 * (pyr_height)))
    # Plot the cum loss
    axes[0].plot(train_history["loss_cum"], label="Train")
    axes[0].plot(train_history["val_loss_cum"], label="Validation")
    axes[0].set_title("Cumulative Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    # Plot the loss in one plot, and the accuracy, precision, and recall in another plot
    for k in range(pyr_height):
        axes[1 + 2 * k].plot(train_history["loss{}".format(k)], label="Train")
        axes[1 + 2 * k].plot(train_history["val_loss{}".format(k)], label="Validation")
        axes[1 + 2 * k].set_title("Loss (Output {})".format(k))
        axes[1 + 2 * k].set_xlabel("Epoch")
        axes[1 + 2 * k].set_ylabel("Loss")
        axes[1 + 2 * k].legend()
        axes[2 + 2 * k].plot(train_history["accuracy{}".format(k)], label="Train Accuracy")
        axes[2 + 2 * k].plot(train_history["val_accuracy{}".format(k)], label="Validation Accuracy")
        axes[2 + 2 * k].plot(train_history["precision{}".format(k)], label="Train Precision")
        axes[2 + 2 * k].plot(train_history["val_precision{}".format(k)], label="Validation Precision")
        axes[2 + 2 * k].plot(train_history["recall{}".format(k)], label="Train Recall")
        axes[2 + 2 * k].plot(train_history["val_recall{}".format(k)], label="Validation Recall")
        axes[2 + 2 * k].set_title("Accuracy, Precision, and Recall (Output {})".format(k))
        axes[2 + 2 * k].set_xlabel("Epoch")
        axes[2 + 2 * k].set_ylabel("Metric")
        axes[2 + 2 * k].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "train_history.png"))