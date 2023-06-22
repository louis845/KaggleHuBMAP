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
import model_multiclass_base
import image_sampling
import image_wsi_sampling


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a progressively supervised U-Net model with reconstructed WSI data.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use. Default 1.")
    parser.add_argument("--augmentation", action="store_true", help="Whether to use data augmentation. Default False.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate to use. Default 1e-5.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true", help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--use_atrous_conv", action="store_true", help="Whether to use atrous convolutional networks. Default False.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")
    parser.add_argument("--unet_attention", action="store_true", help="Whether to use attention in the U-Net. Default False.")
    parser.add_argument("--deep_exponent_base", type=float, default=2.0, help="The base of the exponent for the deep supervision loss. Default 2.0.")
    parser.add_argument("--mixup", type=float, default=0.0, help="The alpha value of mixup. Default 0, meaning no mixup.")
    parser.add_argument("--image_size", type=int, default=1024, help="The size of the images to use. Default 1024.")

    model_data_manager.model_add_argparse_arguments(parser)

    args = parser.parse_args()

    model_dir, dataset_loader, training_entries, validation_entries, prev_model_checkpoint_dir, extra_info, train_subdata, val_subdata = model_data_manager.model_get_argparse_arguments(args, return_subdata_name=True)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    training_entries = np.array(training_entries, dtype=object)
    validation_entries = np.array(validation_entries, dtype=object)
    mixup = args.mixup
    image_size = args.image_size


    if args.unet_attention:
        model = model_unet_attention.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                                    hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                                    use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                    in_channels=4, use_atrous_conv=args.use_atrous_conv,
                                                    deep_supervision=True).to(device=config.device)
    else:
        model = model_unet_base.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                                hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                               use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                               in_channels=4, use_atrous_conv=args.use_atrous_conv, deep_supervision=True).to(device=config.device)

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

    num_epochs = args.epochs
    batch_size = args.batch_size
    augmentation = args.augmentation
    epochs_per_save = args.epochs_per_save
    image_pixels_round = 2 ** args.pyramid_height
    pyr_height = args.pyramid_height
    deep_exponent_base = args.deep_exponent_base

    model_config = {
        "model": "reconstructed_model_progressive_supervised_unet",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "augmentation": augmentation,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "epochs_per_save": epochs_per_save,
        "use_batch_norm": args.use_batch_norm,
        "use_res_conv": args.use_res_conv,
        "use_atrous_conv": args.use_atrous_conv,
        "hidden_channels": args.hidden_channels,
        "pyramid_height": args.pyramid_height,
        "unet_attention": args.unet_attention,
        "deep_exponent_base": deep_exponent_base,
        "mixup": mixup,
        "image_size": args.image_size,
        "training_script": "reconstructed_model_progressive_supervised_unet.py",
    }
    for key, value in extra_info.items():
        model_config[key] = value

    # Initialize training
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

    print("Using class weights:")
    classes = ["blood_vessel", "boundary"]
    class_weights = [1.0, 1.0]
    num_classes = 2
    for k in range(len(classes)):
        seg_class = classes[k]
        train_history["accuracy_{}".format(seg_class)] = []
        train_history["val_accuracy_{}".format(seg_class)] = []
        train_history["precision_{}".format(seg_class)] = []
        train_history["val_precision_{}".format(seg_class)] = []
        train_history["recall_{}".format(seg_class)] = []
        train_history["val_recall_{}".format(seg_class)] = []
        print("Class {}: {}".format(seg_class, class_weights[k]))

    class_weights = torch.tensor([1.0] + class_weights, dtype=torch.float32, device=config.device)

    # Initialize image sampler here
    train_sampler = image_wsi_sampling.get_image_sampler(train_subdata, image_width=image_size)
    val_sampler = image_wsi_sampling.get_image_sampler(val_subdata, image_width=image_size)

    # Training loop
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
        true_negative_class, true_positive_class, false_negative_class, false_positive_class = {}, {}, {}, {}
        for seg_class in classes:
            true_negative_class[seg_class], true_positive_class[seg_class], false_negative_class[seg_class], false_positive_class[seg_class] = 0, 0, 0, 0

        # Shuffle
        training_entries_shuffle = rng.permutation(training_entries)
        if mixup > 0.0:
            training_entries_shuffle2 = rng.permutation(training_entries)

        while trained < len(training_entries):
            batch_end = min(trained + batch_size, len(training_entries))
            batch_indices = training_entries_shuffle[trained:batch_end]
            length = len(batch_indices)
            if mixup > 0.0:
                batch_indices2 = training_entries_shuffle2[trained:batch_end]

                train_image_cat_batch, train_image_ground_truth_batch, train_image_ground_truth_mask_batch, train_image_ground_truth_deep, train_image_ground_truth_mask_deep =\
                    train_sampler.obtain_random_sample_with_mixup_batch(batch_indices, batch_indices2, mixup_alpha=mixup, augmentation=augmentation, deep_supervision_downsamples=pyr_height - 1)

            else:
                train_image_cat_batch, train_image_ground_truth_batch, train_image_ground_truth_mask_batch, train_image_ground_truth_deep, train_image_ground_truth_mask_deep = \
                    train_sampler.obtain_random_sample_batch(batch_indices, augmentation=augmentation, deep_supervision_downsamples=pyr_height - 1)

            gc.collect()
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            result, deep_outputs = model(train_image_cat_batch)

            loss = 0.0
            for k in range(pyr_height - 2, -1, -1):
                multiply_scale_factor = deep_exponent_base ** (pyr_height - 1 - k)

                k_loss = torch.sum(torch.nn.functional.binary_cross_entropy(deep_outputs[k], train_image_ground_truth_deep[pyr_height - 2 - k], reduction="none")
                         * train_image_ground_truth_mask_deep[pyr_height - 2 - k]) * multiply_scale_factor
                loss += k_loss

                total_loss_per_output[k] += k_loss.item()

                with torch.no_grad():
                    deep_class_prediction = torch.argmax(deep_outputs[k], dim=1)
                    if mixup > 0.0:
                        train_image_ground_truth_deep_class = torch.argmax(train_image_ground_truth_deep[pyr_height - 2 - k], dim=1)
                    else:
                        train_image_ground_truth_deep_class = train_image_ground_truth_deep[pyr_height - 2 - k]
                    bool_mask = train_image_ground_truth_mask_deep[k].to(torch.bool)
                    true_negative_per_output[k] += ((deep_class_prediction == 0) & (train_image_ground_truth_deep_class == 0) & bool_mask).sum().item()
                    false_negative_per_output[k] += ((deep_class_prediction == 0) & (train_image_ground_truth_deep_class > 0) & bool_mask).sum().item()
                    true_positive_per_output[k] += ((deep_class_prediction > 0) & (train_image_ground_truth_deep_class > 0) & bool_mask).sum().item()
                    false_positive_per_output[k] += ((deep_class_prediction > 0) & (train_image_ground_truth_deep_class == 0) & bool_mask).sum().item()


            result_loss = torch.sum(torch.nn.functional.cross_entropy(result, train_image_ground_truth_batch, reduction="none", weight=class_weights) * train_image_ground_truth_mask_batch)
            loss += result_loss

            total_loss_per_output[-1] += result_loss.item()
            with torch.no_grad():
                pred_labels = torch.argmax(result, dim=1)
                bool_mask = train_image_ground_truth_mask_batch.to(torch.bool)
                if mixup > 0.0:
                    train_image_ground_truth_batch = torch.argmax(train_image_ground_truth_batch, dim=1)
                true_negative_per_output[-1] += ((pred_labels == 0) & (train_image_ground_truth_batch == 0) & bool_mask).sum().item()
                false_negative_per_output[-1] += ((pred_labels == 0) & (train_image_ground_truth_batch > 0) & bool_mask).sum().item()
                true_positive_per_output[-1] += ((pred_labels > 0) & (train_image_ground_truth_batch > 0) & bool_mask).sum().item()
                false_positive_per_output[-1] += ((pred_labels > 0) & (train_image_ground_truth_batch == 0) & bool_mask).sum().item()

                for seg_idx in range(len(classes)):
                    seg_class = classes[seg_idx]
                    seg_ps = seg_idx + 1
                    true_positive_class[seg_class] += int(torch.sum((pred_labels == seg_ps) & (train_image_ground_truth_batch == seg_ps) & bool_mask).item())
                    true_negative_class[seg_class] += int(torch.sum((pred_labels != seg_ps) & (train_image_ground_truth_batch != seg_ps) & bool_mask).item())
                    false_positive_class[seg_class] += int(torch.sum((pred_labels == seg_ps) & (train_image_ground_truth_batch != seg_ps) & bool_mask).item())
                    false_negative_class[seg_class] += int(torch.sum((pred_labels != seg_ps) & (train_image_ground_truth_batch == seg_ps) & bool_mask).item())

            loss.backward()
            optimizer.step()

            total_cum_loss += loss.item()

            trained += len(batch_indices)

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
            true_negative_class, true_positive_class, false_negative_class, false_positive_class = {}, {}, {}, {}
            for seg_class in classes:
                true_negative_class[seg_class], true_positive_class[seg_class], false_negative_class[seg_class], \
                false_positive_class[seg_class] = 0, 0, 0, 0

            while tested < len(validation_entries):
                batch_end = min(tested + batch_size, len(validation_entries))
                batch_indices = validation_entries[tested:batch_end]

                test_image_cat_batch, test_image_ground_truth_batch, test_image_ground_truth_mask_batch, test_image_ground_truth_deep, test_image_ground_truth_mask_deep = \
                    val_sampler.obtain_random_sample_batch(batch_indices, augmentation=False, deep_supervision_downsamples=pyr_height - 1)

                result, deep_outputs = model(test_image_cat_batch)

                loss = 0.0
                for k in range(pyr_height - 2, -1, -1):
                    multiply_scale_factor = deep_exponent_base ** (pyr_height - 1 - k)

                    k_loss = (torch.nn.functional.cross_entropy(deep_outputs[k], test_image_ground_truth_deep[pyr_height - 2 - k], reduction="none")
                              * test_image_ground_truth_mask_deep[pyr_height - 2 - k]) * multiply_scale_factor
                    loss += k_loss

                    total_loss_per_output[k] += k_loss.item()

                    deep_class_prediction = torch.argmax(deep_outputs[k], dim=1)
                    if mixup > 0.0:
                        test_image_ground_truth_deep_class = torch.argmax(test_image_ground_truth_deep[pyr_height - 2 - k], dim=1)
                    else:
                        test_image_ground_truth_deep_class = test_image_ground_truth_deep[pyr_height - 2 - k]
                    bool_mask = test_image_ground_truth_mask_deep[pyr_height - 2 - k].bool()
                    true_negative_per_output[k] += ((deep_class_prediction == 0) & (test_image_ground_truth_deep_class == 0) & bool_mask).sum().item()
                    false_negative_per_output[k] += ((deep_class_prediction == 0) & (test_image_ground_truth_deep_class == 1) & bool_mask).sum().item()
                    true_positive_per_output[k] += ((deep_class_prediction == 1) & (test_image_ground_truth_deep_class == 1) & bool_mask).sum().item()
                    false_positive_per_output[k] += ((deep_class_prediction == 1) & (test_image_ground_truth_deep_class == 0) & bool_mask).sum().item()

                result_loss = (torch.nn.functional.cross_entropy(result, test_image_ground_truth_batch, reduction="none", weight=class_weights)
                                 * test_image_ground_truth_mask_batch).sum()
                loss += result_loss

                total_loss_per_output[-1] += result_loss.item()

                if mixup > 0.0:
                    test_image_ground_truth_batch = torch.argmax(test_image_ground_truth_batch, dim=1)

                pred_labels = torch.argmax(result, dim=1)
                bool_mask = test_image_ground_truth_mask_batch.to(dtype=torch.bool)
                true_negative_per_output[-1] += ((pred_labels == 0) & (test_image_ground_truth_batch == 0) & bool_mask).sum().item()
                false_negative_per_output[-1] += ((pred_labels == 0) & (test_image_ground_truth_batch > 0) & bool_mask).sum().item()
                true_positive_per_output[-1] += ((pred_labels > 0) & (test_image_ground_truth_batch > 0) & bool_mask).sum().item()
                false_positive_per_output[-1] += ((pred_labels > 0) & (test_image_ground_truth_batch == 0) & bool_mask).sum().item()

                for seg_idx in range(len(classes)):
                    seg_class = classes[seg_idx]
                    seg_ps = seg_idx + 1
                    true_positive_class[seg_class] += int(
                        torch.sum((pred_labels == seg_ps) & (test_image_ground_truth_batch == seg_ps) & bool_mask).item())
                    true_negative_class[seg_class] += int(
                        torch.sum((pred_labels != seg_ps) & (test_image_ground_truth_batch != seg_ps) & bool_mask).item())
                    false_positive_class[seg_class] += int(
                        torch.sum((pred_labels == seg_ps) & (test_image_ground_truth_batch != seg_ps) & bool_mask).item())
                    false_negative_class[seg_class] += int(
                        torch.sum((pred_labels != seg_ps) & (test_image_ground_truth_batch == seg_ps) & bool_mask).item())

                total_cum_loss += loss.item()

                tested += len(batch_indices)

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
        for seg_class in classes:
            print("{} (Accuracy {})".format(train_history["accuracy_" + seg_class][-1], seg_class))
            print("{} (Val Accuracy {})".format(train_history["val_accuracy_" + seg_class][-1], seg_class))
            print("{} (Precision {})".format(train_history["precision_" + seg_class][-1], seg_class))
            print("{} (Val Precision {})".format(train_history["val_precision_" + seg_class][-1], seg_class))
            print("{} (Recall {})".format(train_history["recall_" + seg_class][-1], seg_class))
            print("{} (Val Recall {})".format(train_history["val_recall_" + seg_class][-1], seg_class))
        print("Learning Rate: {}".format(args.learning_rate))
        print("")

        # Save the training history
        train_history_save = pd.DataFrame(train_history)
        train_history_save.to_csv(os.path.join(model_dir, "train_history.csv"), index=False)
        del train_history_save

        ctime = time.time()

        del train_image_cat_batch, train_image_ground_truth_batch, train_image_ground_truth_mask_batch, train_image_ground_truth_deep[:], train_image_ground_truth_mask_deep[:]
        del test_image_cat_batch, test_image_ground_truth_batch, test_image_ground_truth_mask_batch, test_image_ground_truth_deep[:], test_image_ground_truth_mask_deep[:]
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
    fig, axes = plt.subplots(1 + 2 * pyr_height + num_classes, 1, figsize=(12, 4 + 8 * pyr_height + 4 * num_classes))
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

    for k in range(num_classes):
        axes[1 + 2 * pyr_height + k].plot(train_history["accuracy_" + classes[k]], label="Train Accuracy")
        axes[1 + 2 * pyr_height + k].plot(train_history["val_accuracy_" + classes[k]], label="Validation Accuracy")
        axes[1 + 2 * pyr_height + k].plot(train_history["precision_" + classes[k]], label="Train Precision")
        axes[1 + 2 * pyr_height + k].plot(train_history["val_precision_" + classes[k]], label="Validation Precision")
        axes[1 + 2 * pyr_height + k].plot(train_history["recall_" + classes[k]], label="Train Recall")
        axes[1 + 2 * pyr_height + k].plot(train_history["val_recall_" + classes[k]], label="Validation Recall")
        axes[1 + 2 * pyr_height + k].set_title("Accuracy, Precision, and Recall ({})".format(classes[k]))
        axes[1 + 2 * pyr_height + k].set_xlabel("Epoch")
        axes[1 + 2 * pyr_height + k].set_ylabel("Metric")
        axes[1 + 2 * pyr_height + k].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "train_history.png"))