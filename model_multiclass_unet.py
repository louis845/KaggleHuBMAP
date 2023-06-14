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
import model_unet_attention
import model_multiclass_base
import model_simple_unet
import image_sampling

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multiclass U-Net model")
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

    if not os.path.isfile("multiclass_config.json"):
        print("You need to create a multiclass config file \"multiclass_config.json\". A template file is generated for you.")
        model_multiclass_base.generate_multiclass_config("multiclass_config.json")
        exit(-1)
    class_config, classes, num_classes = model_multiclass_base.load_multiclass_config("multiclass_config.json")
    model_multiclass_base.save_multiclass_config(os.path.join(model_dir, "multiclass_config.json"), class_config)

    if args.unet_attention:
        model = model_unet_attention.UNetClassifier(num_classes=num_classes, hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                               use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height, in_channels=args.in_channels, use_atrous_conv=args.use_atrous_conv).to(device=config.device)
    else:
        model = model_unet_base.UNetClassifier(num_classes=num_classes, hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                               use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height, in_channels=args.in_channels, use_atrous_conv=args.use_atrous_conv).to(device=config.device)
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

    # Train the model
    train_history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "precision": [], "val_precision": [], "recall": [], "val_recall": []}
    for seg_class in classes:
        train_history["accuracy_{}".format(seg_class)] = []
        train_history["val_accuracy_{}".format(seg_class)] = []
        train_history["precision_{}".format(seg_class)] = []
        train_history["val_precision_{}".format(seg_class)] = []
        train_history["recall_{}".format(seg_class)] = []
        train_history["val_recall_{}".format(seg_class)] = []

    loss_function = torch.nn.CrossEntropyLoss(reduction="none")

    batch_size = args.batch_size
    num_epochs = args.epochs
    rotation_augmentation = args.rotation_augmentation
    epochs_per_save = args.epochs_per_save
    image_pixels_round = 2 ** args.pyramid_height
    in_channels = args.in_channels
    mixup = args.mixup

    model_config = {
        "model": "model_multiclass_unet",
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
        "mixup": mixup,
        "training_script": "model_multiclass_unet.py",
    }
    for key, value in extra_info.items():
        model_config[key] = value

    for seg_class in classes:
        if seg_class not in dataset_loader.list_segmentation_masks():
            print("Invalid segmentation class in multiclass_config.json. The available options are: {}".format(dataset_loader.list_segmentation_masks()))
            exit(1)

    # Compute the number of positive and negative pixels in the training data
    with torch.no_grad():
        num_background_positive_pixels, num_background_negative_pixels, num_foreground_positive_pixels,\
            num_foreground_negative_pixels, num_dset1_entries = model_simple_unet.compute_background_foreground(training_entries, dataset_loader, "blood_vessel", "blood_vessel")

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

    # Precompute the classes masks (long)
    print("Precomputing the classes masks...")
    class_labels_dict = model_multiclass_base.precompute_classes(dataset_loader, list(training_entries) + list(validation_entries), classes)
    print("Finished precomputing. Training the model now......")

    rng = np.random.default_rng()
    for epoch in range(num_epochs):
        ctime = time.time()
        # Train the model
        # Split the training data into batches
        trained = 0
        total_loss = 0.0
        true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0
        true_negative_class, true_positive_class, false_negative_class, false_positive_class = {}, {}, {}, {}
        for seg_class in classes:
            true_negative_class[seg_class], true_positive_class[seg_class], false_negative_class[seg_class], false_positive_class[seg_class] = 0, 0, 0, 0

        # Shuffle
        training_entries_shuffle = rng.permutation(training_entries)
        if mixup > 0.0:
            training_entries_shuffle2 = rng.permutation(training_entries)

        steps = 0
        while trained < len(training_entries):
            optimizer.zero_grad()

            batch_end = min(trained + batch_size, len(training_entries))
            batch_indices = training_entries_shuffle[trained:batch_end]

            if mixup > 0.0:
                batch_indices2 = training_entries_shuffle2[trained:batch_end]
                train_image_data_batch, train_image_ground_truth_batch, train_image_multiclass_gt_batch, train_image_ground_truth_ds_batch, \
                    train_image_multiclass_gt_ds_batch = image_sampling.sample_images_mixup(batch_indices, batch_indices2, dataset_loader, mixup_alpha=mixup,
                                                                                      rotation_augmentation=rotation_augmentation,
                                                                                      multiclass_labels_dict=class_labels_dict, num_classes=num_classes,
                                                                                      deep_supervision_downsamples=0,
                                                                                      crop_height=512, crop_width=512)
            else:
                train_image_data_batch, train_image_ground_truth_batch, train_image_multiclass_gt_batch, train_image_ground_truth_ds_batch, \
                    train_image_multiclass_gt_ds_batch = image_sampling.sample_images(batch_indices, dataset_loader,
                                                                                      rotation_augmentation=rotation_augmentation,
                                                                                      multiclass_labels_dict=class_labels_dict,
                                                                                      deep_supervision_downsamples=0,
                                                                                      crop_height=512, crop_width=512)

            y_pred = model(train_image_data_batch)

            loss = loss_function(y_pred, train_image_multiclass_gt_batch)
            # Weighted loss, with precomputed weights
            loss = torch.sum(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                y_pred = torch.argmax(y_pred, dim=1)
                if mixup > 0.0:
                    train_image_multiclass_gt_batch = torch.argmax(train_image_multiclass_gt_batch, dim=1)
                true_positive += int(torch.sum((y_pred > 0) & (train_image_multiclass_gt_batch > 0)).item())
                true_negative += int(torch.sum((y_pred == 0) & (train_image_multiclass_gt_batch == 0)).item())
                false_positive += int(torch.sum((y_pred > 0) & (train_image_multiclass_gt_batch == 0)).item())
                false_negative += int(torch.sum((y_pred == 0) & (train_image_multiclass_gt_batch > 0)).item())

                for seg_idx in range(len(classes)):
                    seg_class = classes[seg_idx]
                    seg_ps = seg_idx + 1
                    true_positive_class[seg_class] += int(torch.sum((y_pred == seg_ps) & (train_image_multiclass_gt_batch == seg_ps)).item())
                    true_negative_class[seg_class] += int(torch.sum((y_pred != seg_ps) & (train_image_multiclass_gt_batch != seg_ps)).item())
                    false_positive_class[seg_class] += int(torch.sum((y_pred == seg_ps) & (train_image_multiclass_gt_batch != seg_ps)).item())
                    false_negative_class[seg_class] += int(torch.sum((y_pred != seg_ps) & (train_image_multiclass_gt_batch == seg_ps)).item())

            trained += len(batch_indices)

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

        for seg_class in classes:
            train_history["accuracy_" + seg_class].append((true_positive_class[seg_class] + true_negative_class[seg_class]) / (true_positive_class[seg_class] + true_negative_class[seg_class] + false_positive_class[seg_class] + false_negative_class[seg_class]))
            if true_positive_class[seg_class] + false_positive_class[seg_class] == 0:
                train_history["precision_" + seg_class].append(0.0)
            else:
                train_history["precision_" + seg_class].append(true_positive_class[seg_class] / (true_positive_class[seg_class] + false_positive_class[seg_class]))
            if true_positive_class[seg_class] + false_negative_class[seg_class] == 0:
                train_history["recall_" + seg_class].append(0.0)
            else:
                train_history["recall_" + seg_class].append(true_positive_class[seg_class] / (true_positive_class[seg_class] + false_negative_class[seg_class]))


        # Test the model
        with torch.no_grad():
            tested = 0
            total_loss = 0.0
            true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0
            true_negative_class, true_positive_class, false_negative_class, false_positive_class = {}, {}, {}, {}
            for seg_class in classes:
                true_negative_class[seg_class], true_positive_class[seg_class], false_negative_class[seg_class], false_positive_class[seg_class] = 0, 0, 0, 0
            while tested < len(validation_entries):
                batch_end = min(tested + batch_size, len(validation_entries))
                batch_indices = validation_entries[tested:batch_end]
                test_image_data_batch, test_image_ground_truth_batch, test_image_multiclass_gt_batch, test_image_ground_truth_ds_batch, \
                    test_image_multiclass_gt_ds_batch = image_sampling.sample_images(batch_indices, dataset_loader,
                                                                                     rotation_augmentation=False,
                                                                                     multiclass_labels_dict=class_labels_dict,
                                                                                     deep_supervision_downsamples=0)

                y_pred = model(test_image_data_batch)
                loss = loss_function(y_pred, test_image_multiclass_gt_batch)
                # Weighted loss, with precomputed weights
                loss = torch.sum(loss)
                total_loss += loss.item()

                y_pred = torch.argmax(y_pred, dim=1)
                true_positive += int(torch.sum((y_pred > 0) & (test_image_multiclass_gt_batch > 0)).item())
                true_negative += int(torch.sum((y_pred == 0) & (test_image_multiclass_gt_batch == 0)).item())
                false_positive += int(torch.sum((y_pred > 0) & (test_image_multiclass_gt_batch == 0)).item())
                false_negative += int(torch.sum((y_pred == 0) & (test_image_multiclass_gt_batch > 0)).item())

                for seg_idx in range(len(classes)):
                    seg_class = classes[seg_idx]
                    seg_ps = seg_idx + 1
                    true_positive_class[seg_class] += int(torch.sum((y_pred == seg_ps) & (test_image_multiclass_gt_batch == seg_ps)).item())
                    true_negative_class[seg_class] += int(torch.sum((y_pred != seg_ps) & (test_image_multiclass_gt_batch != seg_ps)).item())
                    false_positive_class[seg_class] += int(torch.sum((y_pred == seg_ps) & (test_image_multiclass_gt_batch != seg_ps)).item())
                    false_negative_class[seg_class] += int(torch.sum((y_pred != seg_ps) & (test_image_multiclass_gt_batch == seg_ps)).item())

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
            for seg_class in classes:
                train_history["val_accuracy_" + seg_class].append((true_positive_class[seg_class] + true_negative_class[seg_class]) / (true_positive_class[seg_class] + true_negative_class[seg_class] + false_positive_class[seg_class] + false_negative_class[seg_class]))
                if true_positive_class[seg_class] + false_positive_class[seg_class] == 0:
                    train_history["val_precision_" + seg_class].append(0.0)
                else:
                    train_history["val_precision_" + seg_class].append(true_positive_class[seg_class] / (true_positive_class[seg_class] + false_positive_class[seg_class]))
                if true_positive_class[seg_class] + false_negative_class[seg_class] == 0:
                    train_history["val_recall_" + seg_class].append(0.0)
                else:
                    train_history["val_recall_" + seg_class].append(true_positive_class[seg_class] / (true_positive_class[seg_class] + false_negative_class[seg_class]))

        print("Time Elapsed: {}".format(time.time() - ctime))
        print("Epoch: {}/{}".format(epoch, num_epochs))
        print("{} (Loss)".format(train_history["loss"][-1]))
        print("{} (Val Loss)".format(train_history["val_loss"][-1]))
        print("{} (Accuracy)".format(train_history["accuracy"][-1]))
        print("{} (Val Accuracy)".format(train_history["val_accuracy"][-1]))
        print("{} (Precision)".format(train_history["precision"][-1]))
        print("{} (Val Precision)".format(train_history["val_precision"][-1]))
        print("{} (Recall)".format(train_history["recall"][-1]))
        print("{} (Val Recall)".format(train_history["val_recall"][-1]))
        for seg_class in classes:
            print("{} (Accuracy {})".format(train_history["accuracy_" + seg_class][-1], seg_class))
            print("{} (Val Accuracy {})".format(train_history["val_accuracy_" + seg_class][-1], seg_class))
            print("{} (Precision {})".format(train_history["precision_" + seg_class][-1], seg_class))
            print("{} (Val Precision {})".format(train_history["val_precision_" + seg_class][-1], seg_class))
            print("{} (Recall {})".format(train_history["recall_" + seg_class][-1], seg_class))
            print("{} (Val Recall {})".format(train_history["val_recall_" + seg_class][-1], seg_class))
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
    fig, axes = plt.subplots(2 + num_classes, 1, figsize=(12, 8 + num_classes * 4))
    axes[0].plot(train_history["loss"], label="Loss")
    axes[0].plot(train_history["val_loss"], label="Val Loss")
    axes[1].plot(train_history["accuracy"], label="Accuracy")
    axes[1].plot(train_history["val_accuracy"], label="Val Accuracy")
    axes[1].plot(train_history["precision"], label="Precision")
    axes[1].plot(train_history["val_precision"], label="Val Precision")
    axes[1].plot(train_history["recall"], label="Recall")
    axes[1].plot(train_history["val_recall"], label="Val Recall")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[0].legend()
    axes[1].legend()

    for seg_idx in range(len(classes)):
        seg_class = classes[seg_idx]
        axes[2 + seg_idx].plot(train_history["accuracy_" + seg_class], label="Accuracy " + seg_class)
        axes[2 + seg_idx].plot(train_history["val_accuracy_" + seg_class], label="Val Accuracy " + seg_class)
        axes[2 + seg_idx].plot(train_history["precision_" + seg_class], label="Precision " + seg_class)
        axes[2 + seg_idx].plot(train_history["val_precision_" + seg_class], label="Val Precision " + seg_class)
        axes[2 + seg_idx].plot(train_history["recall_" + seg_class], label="Recall " + seg_class)
        axes[2 + seg_idx].plot(train_history["val_recall_" + seg_class], label="Val Recall " + seg_class)
        axes[2 + seg_idx].set_xlabel("Epoch")
        axes[2 + seg_idx].set_ylabel("Metric")
        axes[2 + seg_idx].legend()


    plt.savefig(os.path.join(model_dir, "train_history.png"))