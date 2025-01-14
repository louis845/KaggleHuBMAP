import gc
import os
import time
import argparse
import json
import contextlib
import traceback

import tqdm

import config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn
import torch.cuda.amp

import model_data_manager
import model_unet_base
import model_unet_attention
import image_wsi_sampling
import logging_memory_utils

def compute_confidence_class(result: torch.Tensor):
    with torch.no_grad():
        if use_separated_focal_loss:
            return (result[:, 0, ...] <= 0.0) * (torch.argmax(result[:, 1:, ...], dim=1) + 1)
        else:
            softmax = torch.softmax(result, dim=1)
            # (confidence level at least 0.5)
            return (softmax[:, 0, ...] <= 0.5) * (torch.argmax(result[:, 1:, ...], dim=1) + 1)

def compute_class_no_confidence(result: torch.Tensor):
    with torch.no_grad():
        return torch.argmax(result[:, 1:, ...], dim=1) + 1

def focal_loss(result: torch.Tensor, ground_truth: torch.Tensor, one_hot_ground_truth:bool):
    if use_separated_focal_loss:
        mclass_weights = class_weights_old
    else:
        mclass_weights = class_weights
    cross_entropy = torch.nn.functional.cross_entropy(result, ground_truth, reduction="none", weight=mclass_weights)

    ground_truth_one_hot = ground_truth if one_hot_ground_truth else torch.nn.functional.one_hot(ground_truth, num_classes=3).permute(0, 3, 1, 2).to(torch.float32)
    softmax = torch.softmax(result, dim=1)

    return torch.sum((softmax - ground_truth_one_hot) ** 2, dim=1) * cross_entropy

def math_focal_loss(result: torch.Tensor, ground_truth: torch.Tensor, one_hot_ground_truth:bool):
    if use_separated_focal_loss:
        mclass_weights = class_weights_old
    else:
        mclass_weights = class_weights
    cross_entropy = torch.nn.functional.cross_entropy(result, ground_truth, reduction="none")

    with torch.no_grad():
        ground_truth_one_hot = ground_truth if one_hot_ground_truth else torch.nn.functional.one_hot(ground_truth, num_classes=3).permute(0, 3, 1, 2).to(torch.float32)
        weights = torch.sum(ground_truth_one_hot * mclass_weights.unsqueeze(-1).unsqueeze(-1), dim=1)

    softmax = torch.softmax(result, dim=1)

    return weights * torch.sum((softmax - ground_truth_one_hot) ** 2, dim=1) * cross_entropy

def composite_focal_loss(result: torch.Tensor, ground_truth: torch.Tensor, one_hot_ground_truth:bool):
    ground_truth_one_hot = ground_truth if one_hot_ground_truth else torch.nn.functional.one_hot(ground_truth, num_classes=3).permute(0, 3, 1, 2).to(torch.float32)
    with torch.no_grad():
        ground_truth_other = torch.stack([ground_truth_one_hot[:, 0, :, :] + ground_truth_one_hot[:, 1, :, :], ground_truth_one_hot[:, 2, :, :]], dim=1)
        non_backgroundness = 1 - ground_truth_one_hot[:, 0, :, :]

    cross_entropy = torch.nn.functional.cross_entropy(result, ground_truth, reduction="none", weight=class_weights)
    cross_entropy_cp = torch.nn.functional.cross_entropy(result[:, 1:, :, :], ground_truth_other, reduction="none", weight=class_weights_composite)
    softmax = torch.softmax(result, dim=1)
    softmax_cp = torch.softmax(result[:, 1:, :, :], dim=1)

    return torch.sum((softmax - ground_truth_one_hot) ** 2, dim=1) * cross_entropy +\
        non_backgroundness * torch.sum((softmax_cp - ground_truth_other) ** 2, dim=1) * cross_entropy_cp

def separated_focal_loss(result: torch.Tensor, ground_truth: torch.Tensor, one_hot_ground_truth:bool):
    with torch.no_grad():
        ground_truth_one_hot = ground_truth if one_hot_ground_truth else torch.nn.functional.one_hot(ground_truth, num_classes=3).permute(0, 3, 1, 2).to(torch.float32)
        ground_truth_other = torch.stack([ground_truth_one_hot[:, 0, :, :] + ground_truth_one_hot[:, 1, :, :], ground_truth_one_hot[:, 2, :, :]], dim=1)
        if args.use_heavy_boundary_in_separated_loss:
            non_backgroundness = (1 - ground_truth_one_hot[:, 0, :, :]) * 5.9 + 0.1

        if mathematical:
            bce_weights = class_weights + (1 - class_weights) * ground_truth_one_hot[:, 0, :, :]
            if args.use_heavy_boundary_in_separated_loss:
                ce_weights = torch.sum(ground_truth_other * class_weights_composite.unsqueeze(-1).unsqueeze(-1), dim=1) * non_backgroundness
            else:
                ce_weights = torch.sum(ground_truth_other * class_weights_composite.unsqueeze(-1).unsqueeze(-1), dim=1)

    sigmoid = torch.sigmoid(result[:, 0, :, :])
    softmax = torch.softmax(result[:, 1:, :, :], dim=1)
    if mathematical:
        binary_ce = torch.nn.functional.binary_cross_entropy_with_logits(result[:, 0, :, :], ground_truth_one_hot[:, 0, :, :], reduction="none")
        cross_entropy_boundary = torch.nn.functional.cross_entropy(result[:, 1:, :, :], ground_truth_other, reduction="none")

        return bce_weights * ((sigmoid - ground_truth_one_hot[:, 0, :, :]) ** 2) * binary_ce + \
            ce_weights * torch.sum((softmax - ground_truth_other) ** 2, dim=1) * cross_entropy_boundary
    else:
        binary_ce = torch.nn.functional.binary_cross_entropy_with_logits(result[:, 0, :, :], ground_truth_one_hot[:, 0, :, :], reduction="none", pos_weight=(1 / class_weights))
        cross_entropy_boundary = torch.nn.functional.cross_entropy(result[:, 1:, :, :], ground_truth_other, reduction="none", weight=class_weights_composite)

        if args.use_heavy_boundary_in_separated_loss:
            return class_weights * ((sigmoid - ground_truth_one_hot[:, 0, :, :]) ** 2) * binary_ce + \
                non_backgroundness * torch.sum((softmax - ground_truth_other) ** 2, dim=1) * cross_entropy_boundary
        else:
            return class_weights * ((sigmoid - ground_truth_one_hot[:, 0, :, :]) ** 2) * binary_ce +\
                    torch.sum((softmax - ground_truth_other) ** 2, dim=1) * cross_entropy_boundary

def single_training_step(model_, optimizer_, train_image_cat_batch_, train_image_ground_truth_batch_, train_image_ground_truth_mask_batch_,
                         train_image_ground_truth_deep_, train_image_ground_truth_mask_deep_, mixup_used:bool, use_amp_=False, scaler_=None):
    total_loss_per_outputs = [0] * (pyr_height - 1)
    optimizer_.zero_grad()

    with torch.cuda.amp.autocast(dtype=torch.float16) if use_amp_ else contextlib.nullcontext() as amp_ctx:
        result, deep_outputs = model_(train_image_cat_batch_)

        loss = 0.0
        for k in range(pyr_height - 2, -1, -1):
            multiply_scale_factor = deep_exponent_base ** (pyr_height - 1 - k)
            if (k > 0) and use_suppressed_deepsupervision:
                multiply_scale_factor = 0.0
            elif use_partially_suppressed_deepsupervision:
                multiply_scale_factor /= 20.0

            if use_focal_loss or use_separated_focal_loss or use_composite_focal_loss:
                if mathematical:
                    ce_res = math_focal_loss(deep_outputs[k], train_image_ground_truth_deep_[pyr_height - 2 - k],
                                        one_hot_ground_truth=mixup_used)
                else:
                    ce_res = focal_loss(deep_outputs[k], train_image_ground_truth_deep_[pyr_height - 2 - k],
                                        one_hot_ground_truth=mixup_used)
            else:
                ce_res = torch.nn.functional.cross_entropy(deep_outputs[k],
                                                           train_image_ground_truth_deep_[pyr_height - 2 - k],
                                                           reduction="none")
            k_loss = torch.sum(ce_res * train_image_ground_truth_mask_deep_[pyr_height - 2 - k]) * multiply_scale_factor
            loss = loss + k_loss

            total_loss_per_outputs[k] = k_loss.item()

        if use_separated_focal_loss:
            ce_res = separated_focal_loss(result, train_image_ground_truth_batch_, one_hot_ground_truth=mixup_used)
        elif use_composite_focal_loss:
            ce_res = composite_focal_loss(result, train_image_ground_truth_batch_, one_hot_ground_truth=mixup_used)
        elif use_focal_loss:
            if mathematical:
                ce_res = math_focal_loss(result, train_image_ground_truth_batch_, one_hot_ground_truth=mixup_used)
            else:
                ce_res = focal_loss(result, train_image_ground_truth_batch_, one_hot_ground_truth=mixup_used)
        else:
            ce_res = torch.nn.functional.cross_entropy(result, train_image_ground_truth_batch_, reduction="none",
                                                       weight=class_weights)
        result_loss = torch.sum(ce_res * train_image_ground_truth_mask_batch_)
        loss = loss + result_loss

    if use_amp:
        # scaled loss to avoid overflow
        scaler_.scale(loss).backward()
        scaler_.step(optimizer_)
        scaler_.update()
    else:
        # usual backward pass if not using AMP
        loss.backward()
        if args.gradient_clipping > 0.0:
            try:
                torch.nn.utils.clip_grad_norm_(model_.parameters(), args.gradient_clipping, error_if_nonfinite=True)
                optimizer_.step()
            except RuntimeError as e:
                traceback.print_exc()
                print("Gradient clipping failed, skipping step.")
                optimizer_.zero_grad()
        else:
            optimizer_.step()

    return loss.item(), result_loss.item(), total_loss_per_outputs, result.detach(),\
        [deep_output.detach() for deep_output in deep_outputs]

def training_step(train_history=None):
    # pass train_history = None if extra steps, pass train_history = train_history main step.
    trained = 0
    if train_history is not None:
        total_cum_loss = 0.0
        total_loss_per_output = np.zeros(pyr_height, dtype=np.float64)
        true_negative_per_output = np.zeros(pyr_height, dtype=np.int64)
        false_negative_per_output = np.zeros(pyr_height, dtype=np.int64)
        true_positive_per_output = np.zeros(pyr_height, dtype=np.int64)
        false_positive_per_output = np.zeros(pyr_height, dtype=np.int64)

        true_negative_confidence, true_positive_confidence, false_negative_confidence, false_positive_confidence = 0, 0, 0, 0

        true_negative_class, true_positive_class, false_negative_class, false_positive_class = {}, {}, {}, {}
        true_negative_confidence_class, true_positive_confidence_class, false_negative_confidence_class, false_positive_confidence_class = {}, {}, {}, {}
        true_negative_class_class, true_positive_class_class, false_negative_class_class, false_positive_class_class = {}, {}, {}, {}
        for seg_class in classes:
            true_negative_class[seg_class], true_positive_class[seg_class], false_negative_class[seg_class], \
                false_positive_class[seg_class] = 0, 0, 0, 0
            true_negative_confidence_class[seg_class], true_positive_confidence_class[seg_class], \
                false_negative_confidence_class[seg_class], false_positive_confidence_class[seg_class] = 0, 0, 0, 0
            true_negative_class_class[seg_class], true_positive_class_class[seg_class],\
                false_negative_class_class[seg_class], false_positive_class_class[seg_class] = 0, 0, 0, 0

    # Shuffle if not async. if async its loaded in another process
    if use_async_sampling == 0:
        training_entries_shuffle = rng.permutation(training_entries)
        if mixup > 0.0:
            training_entries_shuffle2 = rng.permutation(training_entries)
            if extra_training_entries is not None:
                extra1_size = num_extra_trains // 2
                extra2_size = num_extra_trains - extra1_size
                training_entries_shuffle = np.concatenate((training_entries_shuffle,
                                                           rng.choice(extra_training_entries, size=num_extra_trains,
                                                                      replace=False)))
                training_entries_shuffle2 = np.concatenate((training_entries_shuffle2,
                                                            rng.choice(training_entries, size=extra1_size,
                                                                       replace=False)))
                training_entries_shuffle2 = np.concatenate((training_entries_shuffle2,
                                                            rng.choice(extra_training_entries, size=extra2_size,
                                                                       replace=False)))

                perm = rng.permutation(np.arange(len(training_entries_shuffle)))
                training_entries_shuffle = training_entries_shuffle[perm]
                training_entries_shuffle2 = training_entries_shuffle2[perm]

        elif extra_training_entries is not None:
                training_entries_shuffle = np.concatenate((training_entries_shuffle, rng.choice(
                    extra_training_entries, size=num_extra_trains, replace=False)))
                training_entries_shuffle = rng.permutation(training_entries_shuffle)
    else:
        if train_history is None:
            async_request_images_train()
        else:
            async_request_images_val()  # else we preload the validation samples in another process. we are in main step

    if mixup_interlace:
        trained = 0
        trained_interlace = 0
        interlace_to_train = len(training_entries) * mixup_interlace_extras
        total_stuff_to_train = total_training_len + interlace_to_train
    else:
        total_stuff_to_train = total_training_len
    with tqdm.tqdm(total=total_stuff_to_train) as pbar:
        if mixup_interlace:
            condition = (trained < total_training_len) or (trained_interlace < interlace_to_train)
        else:
            condition = trained < total_training_len
        while condition:
            if mixup_interlace: # note that mixup interlace can only be used with async sampling.
                if trained_interlace == interlace_to_train:
                    batch_end = min(trained + batch_size, total_training_len)
                    length = batch_end - trained
                    train_version = "NORMAL"
                    mixup_used = mixup > 0.0
                elif trained == total_training_len:
                    batch_end = min(trained_interlace + batch_size, interlace_to_train)
                    length = batch_end - trained_interlace
                    train_version = "INTERLACE"
                    mixup_used = False
                else:
                    # random
                    remaining_normal = total_training_len - trained
                    remaining_interlace = interlace_to_train - trained_interlace
                    assert remaining_normal > 0 and remaining_interlace > 0, "remaining normal or interlace is 0"
                    remaining_total = remaining_normal + remaining_interlace
                    if rng.uniform(low=0.0, high=1.0) <= remaining_normal / remaining_total:
                        batch_end = min(trained + batch_size, total_training_len)
                        length = batch_end - trained
                        train_version = "NORMAL"
                        mixup_used = mixup > 0.0
                    else:
                        batch_end = min(trained_interlace + batch_size, interlace_to_train)
                        length = batch_end - trained_interlace
                        train_version = "INTERLACE"
                        mixup_used = False
            else:
                batch_end = min(trained + batch_size, total_training_len)
                length = batch_end - trained
                mixup_used = mixup > 0.0
            # obtain batch here
            if use_async_sampling != 0:
                if mixup_interlace:
                    if train_version == "NORMAL":
                        train_image_cat_batch, train_image_ground_truth_batch, train_image_ground_truth_mask_batch, train_image_ground_truth_deep, train_image_ground_truth_mask_deep = \
                            train_sampler.get_samples(device=config.device, length=length)
                    else:
                        train_image_cat_batch, train_image_ground_truth_batch, train_image_ground_truth_mask_batch, train_image_ground_truth_deep, train_image_ground_truth_mask_deep = \
                            train_sampler_interlace_nomixup.get_samples(device=config.device, length=length)
                else:
                    train_image_cat_batch, train_image_ground_truth_batch, train_image_ground_truth_mask_batch, train_image_ground_truth_deep, train_image_ground_truth_mask_deep = \
                        train_sampler.get_samples(device=config.device, length=length)
            else:
                batch_indices = training_entries_shuffle[trained:batch_end]
                if mixup > 0.0:
                    batch_indices2 = training_entries_shuffle2[trained:batch_end]

                    train_image_cat_batch, train_image_ground_truth_batch, train_image_ground_truth_mask_batch, train_image_ground_truth_deep, train_image_ground_truth_mask_deep = \
                        train_sampler.obtain_random_sample_with_mixup_batch(batch_indices, batch_indices2,
                                                                            mixup_alpha=mixup,
                                                                            augmentation=augmentation,
                                                                            deep_supervision_downsamples=pyr_height - 1)

                else:
                    train_image_cat_batch, train_image_ground_truth_batch, train_image_ground_truth_mask_batch, train_image_ground_truth_deep, train_image_ground_truth_mask_deep = \
                        train_sampler.obtain_random_sample_batch(batch_indices, augmentation=augmentation,
                                                                 deep_supervision_downsamples=pyr_height - 1)

            #gc.collect()
            #torch.cuda.empty_cache()

            # training step, forward+backward pass+save some metrics and results
            loss, result_loss, total_loss_per_outputs, result, deep_outputs = \
                single_training_step_compile(model, optimizer, train_image_cat_batch, train_image_ground_truth_batch,
                                train_image_ground_truth_mask_batch, train_image_ground_truth_deep,
                                train_image_ground_truth_mask_deep, use_amp_=use_amp, scaler_=scaler if use_amp else None, mixup_used=mixup_used)
            #gc.collect()
            #torch.cuda.empty_cache()

            # compute metrics
            if train_history is not None:
                # append loss metrics, they do not require torch (tensor) computations anymore.
                for k in range(pyr_height - 2, -1, -1):
                    total_loss_per_output[k] += total_loss_per_outputs[k]
                total_loss_per_output[-1] += result_loss  # last level loss
                total_cum_loss += loss  # total cumulative loss

                with torch.no_grad():
                    # compute per level metrics.
                    for k in range(pyr_height - 2, -1, -1):
                        deep_class_prediction = torch.argmax(deep_outputs[k], dim=1)
                        if mixup_used:
                            train_image_ground_truth_deep_class = torch.argmax(
                                train_image_ground_truth_deep[pyr_height - 2 - k], dim=1)
                        else:
                            train_image_ground_truth_deep_class = train_image_ground_truth_deep[pyr_height - 2 - k]
                        bool_mask = train_image_ground_truth_mask_deep[pyr_height - 2 - k].to(torch.bool)
                        true_negative_per_output[k] += ((deep_class_prediction == 0) & (
                                train_image_ground_truth_deep_class == 0) & bool_mask).sum().item()
                        false_negative_per_output[k] += ((deep_class_prediction == 0) & (
                                train_image_ground_truth_deep_class > 0) & bool_mask).sum().item()
                        true_positive_per_output[k] += ((deep_class_prediction > 0) & (
                                train_image_ground_truth_deep_class > 0) & bool_mask).sum().item()
                        false_positive_per_output[k] += ((deep_class_prediction > 0) & (
                                train_image_ground_truth_deep_class == 0) & bool_mask).sum().item()

                        del deep_class_prediction, bool_mask, train_image_ground_truth_deep_class

                    # compute final result metrics
                    pred_labels = torch.argmax(result, dim=1)
                    pred_labels_confidence = compute_confidence_class(result)
                    pred_labels_class = compute_class_no_confidence(result)
                    bool_mask = train_image_ground_truth_mask_batch.to(torch.bool)
                    if mixup_used:
                        train_image_ground_truth_batch_ev = torch.argmax(train_image_ground_truth_batch, dim=1)
                    else:
                        train_image_ground_truth_batch_ev = train_image_ground_truth_batch
                    true_negative_per_output[-1] += (
                            (pred_labels == 0) & (train_image_ground_truth_batch_ev == 0) & bool_mask).sum().item()
                    false_negative_per_output[-1] += (
                            (pred_labels == 0) & (train_image_ground_truth_batch_ev > 0) & bool_mask).sum().item()
                    true_positive_per_output[-1] += (
                            (pred_labels > 0) & (train_image_ground_truth_batch_ev > 0) & bool_mask).sum().item()
                    false_positive_per_output[-1] += (
                            (pred_labels > 0) & (train_image_ground_truth_batch_ev == 0) & bool_mask).sum().item()

                    true_negative_confidence += ((pred_labels_confidence == 0) & (
                                train_image_ground_truth_batch_ev == 0) & bool_mask).sum().item()
                    false_negative_confidence += ((pred_labels_confidence == 0) & (
                                train_image_ground_truth_batch_ev > 0) & bool_mask).sum().item()
                    true_positive_confidence += ((pred_labels_confidence > 0) & (
                                train_image_ground_truth_batch_ev > 0) & bool_mask).sum().item()
                    false_positive_confidence += ((pred_labels_confidence > 0) & (
                                train_image_ground_truth_batch_ev == 0) & bool_mask).sum().item()

                    for seg_idx in range(len(classes)):
                        seg_class = classes[seg_idx]
                        seg_ps = seg_idx + 1
                        true_positive_class[seg_class] += int(torch.sum(
                            (pred_labels == seg_ps) & (train_image_ground_truth_batch_ev == seg_ps) & bool_mask).item())
                        true_negative_class[seg_class] += int(torch.sum(
                            (pred_labels != seg_ps) & (train_image_ground_truth_batch_ev != seg_ps) & bool_mask).item())
                        false_positive_class[seg_class] += int(torch.sum(
                            (pred_labels == seg_ps) & (train_image_ground_truth_batch_ev != seg_ps) & bool_mask).item())
                        false_negative_class[seg_class] += int(torch.sum(
                            (pred_labels != seg_ps) & (train_image_ground_truth_batch_ev == seg_ps) & bool_mask).item())

                        true_positive_confidence_class[seg_class] += int(torch.sum(
                            (pred_labels_confidence == seg_ps) & (
                                        train_image_ground_truth_batch_ev == seg_ps) & bool_mask).item())
                        true_negative_confidence_class[seg_class] += int(torch.sum(
                            (pred_labels_confidence != seg_ps) & (
                                        train_image_ground_truth_batch_ev != seg_ps) & bool_mask).item())
                        false_positive_confidence_class[seg_class] += int(torch.sum(
                            (pred_labels_confidence == seg_ps) & (
                                        train_image_ground_truth_batch_ev != seg_ps) & bool_mask).item())
                        false_negative_confidence_class[seg_class] += int(torch.sum(
                            (pred_labels_confidence != seg_ps) & (
                                        train_image_ground_truth_batch_ev == seg_ps) & bool_mask).item())

                        true_positive_class_class[seg_class] += int(torch.sum(
                            (pred_labels_class == seg_ps) & (
                                        train_image_ground_truth_batch_ev == seg_ps) & bool_mask).item())
                        true_negative_class_class[seg_class] += int(torch.sum(
                            (pred_labels_class != seg_ps) & (
                                        train_image_ground_truth_batch_ev != seg_ps) & bool_mask).item())
                        false_positive_class_class[seg_class] += int(torch.sum(
                            (pred_labels_class == seg_ps) & (
                                        train_image_ground_truth_batch_ev != seg_ps) & bool_mask).item())
                        false_negative_class_class[seg_class] += int(torch.sum(
                            (pred_labels_class != seg_ps) & (
                                        train_image_ground_truth_batch_ev == seg_ps) & bool_mask).item())

                    del pred_labels, pred_labels_confidence, pred_labels_class, bool_mask, train_image_ground_truth_batch_ev

            if mixup_interlace:
                if train_version == "NORMAL":
                    trained += length
                else:
                    trained_interlace += length
            else:
                trained += length

            del train_image_cat_batch, train_image_ground_truth_batch, train_image_ground_truth_mask_batch, \
                train_image_ground_truth_deep[:], train_image_ground_truth_mask_deep[:]
            del result, deep_outputs[:]
            del train_image_ground_truth_deep, train_image_ground_truth_mask_deep, deep_outputs
            #gc.collect()
            #torch.cuda.empty_cache()

            pbar.update(length)
            if mixup_interlace:
                condition = (trained < total_training_len) or (trained_interlace < interlace_to_train)
            else:
                condition = trained < total_training_len

    if train_history is not None:
        total_loss_per_output /= total_stuff_to_train
        total_cum_loss /= total_stuff_to_train

        accuracy_per_output = (true_positive_per_output + true_negative_per_output).astype(np.float64) / (
                    true_positive_per_output + true_negative_per_output + false_positive_per_output + false_negative_per_output)
        precision_per_output = true_positive_per_output.astype(np.float64) / (
                    true_positive_per_output + false_positive_per_output)
        recall_per_output = true_positive_per_output.astype(np.float64) / (
                    true_positive_per_output + false_negative_per_output)

        accuracy_per_output = np.nan_to_num(accuracy_per_output, nan=0.0, posinf=0.0, neginf=0.0)
        precision_per_output = np.nan_to_num(precision_per_output, nan=0.0, posinf=0.0, neginf=0.0)
        recall_per_output = np.nan_to_num(recall_per_output, nan=0.0, posinf=0.0, neginf=0.0)

        for seg_class in classes:
            train_history["accuracy_" + seg_class].append(
                (true_positive_class[seg_class] + true_negative_class[seg_class]) / (
                            true_positive_class[seg_class] + true_negative_class[seg_class] + false_positive_class[
                        seg_class] + false_negative_class[seg_class]))
            if true_positive_class[seg_class] + false_positive_class[seg_class] == 0:
                train_history["precision_" + seg_class].append(0.0)
            else:
                train_history["precision_" + seg_class].append(
                    true_positive_class[seg_class] / (true_positive_class[seg_class] + false_positive_class[seg_class]))
            if true_positive_class[seg_class] + false_negative_class[seg_class] == 0:
                train_history["recall_" + seg_class].append(0.0)
            else:
                train_history["recall_" + seg_class].append(
                    true_positive_class[seg_class] / (true_positive_class[seg_class] + false_negative_class[seg_class]))

        train_history["loss_cum"].append(total_cum_loss)
        for k in range(pyr_height):
            train_history["loss{}".format(k)].append(total_loss_per_output[k])
            train_history["accuracy{}".format(k)].append(accuracy_per_output[k])
            train_history["precision{}".format(k)].append(precision_per_output[k])
            train_history["recall{}".format(k)].append(recall_per_output[k])

        confidence_accuracy = (true_positive_confidence + true_negative_confidence) / (
                    true_positive_confidence + true_negative_confidence + false_positive_confidence + false_negative_confidence)
        if true_positive_confidence + false_positive_confidence == 0:
            confidence_precision = 0.0
        else:
            confidence_precision = true_positive_confidence / (true_positive_confidence + false_positive_confidence)
        if true_positive_confidence + false_negative_confidence == 0:
            confidence_recall = 0.0
        else:
            confidence_recall = true_positive_confidence / (true_positive_confidence + false_negative_confidence)

        train_history["confidence_accuracy"].append(confidence_accuracy)
        train_history["confidence_precision"].append(confidence_precision)
        train_history["confidence_recall"].append(confidence_recall)

        for seg_class in classes:
            train_history["confidence_accuracy_" + seg_class].append(
                (true_positive_confidence_class[seg_class] + true_negative_confidence_class[seg_class]) / (
                            true_positive_confidence_class[seg_class] + true_negative_confidence_class[seg_class] + false_positive_confidence_class[
                        seg_class] + false_negative_confidence_class[seg_class]))
            if true_positive_confidence_class[seg_class] + false_positive_confidence_class[seg_class] == 0:
                train_history["confidence_precision_" + seg_class].append(0.0)
            else:
                train_history["confidence_precision_" + seg_class].append(
                    true_positive_confidence_class[seg_class] / (
                                true_positive_confidence_class[seg_class] + false_positive_confidence_class[seg_class]))
            if true_positive_confidence_class[seg_class] + false_negative_confidence_class[seg_class] == 0:
                train_history["confidence_recall_" + seg_class].append(0.0)
            else:
                train_history["confidence_recall_" + seg_class].append(
                    true_positive_confidence_class[seg_class] / (
                                true_positive_confidence_class[seg_class] + false_negative_confidence_class[seg_class]))

            train_history["class_accuracy_" + seg_class].append(
                (true_positive_class_class[seg_class] + true_negative_class_class[seg_class]) / (
                            true_positive_class_class[seg_class] + true_negative_class_class[seg_class] +
                            false_positive_class_class[seg_class] + false_negative_class_class[seg_class]) )
            if true_positive_class_class[seg_class] + false_positive_class_class[seg_class] == 0:
                train_history["class_precision_" + seg_class].append(0.0)
            else:
                train_history["class_precision_" + seg_class].append(
                    true_positive_class_class[seg_class] / (
                                true_positive_class_class[seg_class] + false_positive_class_class[seg_class]))
            if true_positive_class_class[seg_class] + false_negative_class_class[seg_class] == 0:
                train_history["class_recall_" + seg_class].append(0.0)
            else:
                train_history["class_recall_" + seg_class].append(
                    true_positive_class_class[seg_class] / (
                            true_positive_class_class[seg_class] + false_negative_class_class[seg_class]
                    ))
    gc.collect()
    #torch.cuda.empty_cache()

def validation_step(train_history):
    with torch.no_grad():
        tested = 0
        total_cum_loss = 0.0
        total_loss_per_output = np.zeros(pyr_height, dtype=np.float64)
        true_negative_per_output = np.zeros(pyr_height, dtype=np.int64)
        false_negative_per_output = np.zeros(pyr_height, dtype=np.int64)
        true_positive_per_output = np.zeros(pyr_height, dtype=np.int64)
        false_positive_per_output = np.zeros(pyr_height, dtype=np.int64)

        true_negative_confidence, true_positive_confidence, false_negative_confidence, false_positive_confidence = 0, 0, 0, 0

        true_negative_class, true_positive_class, false_negative_class, false_positive_class = {}, {}, {}, {}
        true_negative_confidence_class, true_positive_confidence_class, false_negative_confidence_class, false_positive_confidence_class = {}, {}, {}, {}
        true_negative_class_class, true_positive_class_class, false_negative_class_class, false_positive_class_class = {}, {}, {}, {}
        for seg_class in classes:
            true_negative_class[seg_class], true_positive_class[seg_class], false_negative_class[seg_class], \
                false_positive_class[seg_class] = 0, 0, 0, 0
            true_negative_confidence_class[seg_class], true_positive_confidence_class[seg_class], \
                false_negative_confidence_class[seg_class], false_positive_confidence_class[seg_class] = 0, 0, 0, 0
            true_negative_class_class[seg_class], true_positive_class_class[seg_class], false_negative_class_class[
                seg_class], \
                false_positive_class_class[seg_class] = 0, 0, 0, 0

        print()
        print("Validating.....")
        print()
        with tqdm.tqdm(total=len(validation_entries)) as pbar:
            while tested < len(validation_entries):
                batch_end = min(tested + batch_size, len(validation_entries))
                batch_indices = validation_entries[tested:batch_end]

                if use_async_sampling == 0:
                    test_image_cat_batch, test_image_ground_truth_batch, test_image_ground_truth_mask_batch, test_image_ground_truth_deep, test_image_ground_truth_mask_deep = \
                        val_sampler.obtain_random_sample_batch(batch_indices, augmentation=False,
                                                               deep_supervision_downsamples=pyr_height - 1,
                                                               random_location=False)
                else:
                    test_image_cat_batch, test_image_ground_truth_batch, test_image_ground_truth_mask_batch, test_image_ground_truth_deep, test_image_ground_truth_mask_deep = \
                        val_sampler.get_samples(device=config.device, length=len(batch_indices))

                result, deep_outputs = model(test_image_cat_batch)

                loss = 0.0
                for k in range(pyr_height - 2, -1, -1):
                    multiply_scale_factor = deep_exponent_base ** (pyr_height - 1 - k)
                    if (k > 0) and use_suppressed_deepsupervision:
                        multiply_scale_factor = 0.0
                    elif use_partially_suppressed_deepsupervision:
                        multiply_scale_factor /= 20.0

                    if use_focal_loss or use_separated_focal_loss or use_composite_focal_loss:
                        if mathematical:
                            ce_res = math_focal_loss(deep_outputs[k], test_image_ground_truth_deep[pyr_height - 2 - k],
                                                one_hot_ground_truth=False)
                        else:
                            ce_res = focal_loss(deep_outputs[k], test_image_ground_truth_deep[pyr_height - 2 - k],
                                                one_hot_ground_truth=False)
                    else:
                        ce_res = torch.nn.functional.cross_entropy(deep_outputs[k],
                                                                   test_image_ground_truth_deep[pyr_height - 2 - k],
                                                                   reduction="none")

                    k_loss = torch.sum(
                        ce_res * test_image_ground_truth_mask_deep[pyr_height - 2 - k]) * multiply_scale_factor
                    del ce_res
                    loss += k_loss

                    total_loss_per_output[k] += k_loss.item()

                    deep_class_prediction = torch.argmax(deep_outputs[k], dim=1)
                    test_image_ground_truth_deep_class = test_image_ground_truth_deep[pyr_height - 2 - k]
                    bool_mask = test_image_ground_truth_mask_deep[pyr_height - 2 - k].bool()
                    true_negative_per_output[k] += ((deep_class_prediction == 0) & (
                                test_image_ground_truth_deep_class == 0) & bool_mask).sum().item()
                    false_negative_per_output[k] += ((deep_class_prediction == 0) & (
                                test_image_ground_truth_deep_class > 0) & bool_mask).sum().item()
                    true_positive_per_output[k] += ((deep_class_prediction > 0) & (
                                test_image_ground_truth_deep_class > 0) & bool_mask).sum().item()
                    false_positive_per_output[k] += ((deep_class_prediction > 0) & (
                                test_image_ground_truth_deep_class == 0) & bool_mask).sum().item()

                    del bool_mask, deep_class_prediction, test_image_ground_truth_deep_class

                if use_separated_focal_loss:
                    ce_res = separated_focal_loss(result, test_image_ground_truth_batch, one_hot_ground_truth=False)
                elif use_composite_focal_loss:
                    ce_res = composite_focal_loss(result, test_image_ground_truth_batch, one_hot_ground_truth=False)
                elif use_focal_loss:
                    if mathematical:
                        ce_res = math_focal_loss(result, test_image_ground_truth_batch, one_hot_ground_truth=False)
                    else:
                        ce_res = focal_loss(result, test_image_ground_truth_batch, one_hot_ground_truth=False)
                else:
                    ce_res = torch.nn.functional.cross_entropy(result, test_image_ground_truth_batch, reduction="none",
                                                               weight=class_weights)
                result_loss = torch.sum(ce_res * test_image_ground_truth_mask_batch)
                del ce_res
                loss += result_loss

                total_loss_per_output[-1] += result_loss.item()

                pred_labels = torch.argmax(result, dim=1)
                pred_labels_confidence = compute_confidence_class(result)
                pred_labels_class = compute_class_no_confidence(result)

                bool_mask = test_image_ground_truth_mask_batch.to(dtype=torch.bool)
                true_negative_per_output[-1] += (
                            (pred_labels == 0) & (test_image_ground_truth_batch == 0) & bool_mask).sum().item()
                false_negative_per_output[-1] += (
                            (pred_labels == 0) & (test_image_ground_truth_batch > 0) & bool_mask).sum().item()
                true_positive_per_output[-1] += (
                            (pred_labels > 0) & (test_image_ground_truth_batch > 0) & bool_mask).sum().item()
                false_positive_per_output[-1] += (
                            (pred_labels > 0) & (test_image_ground_truth_batch == 0) & bool_mask).sum().item()

                true_negative_confidence += ((pred_labels_confidence == 0) & (
                            test_image_ground_truth_batch == 0) & bool_mask).sum().item()
                false_negative_confidence += ((pred_labels_confidence == 0) & (
                            test_image_ground_truth_batch > 0) & bool_mask).sum().item()
                true_positive_confidence += (
                            (pred_labels_confidence > 0) & (test_image_ground_truth_batch > 0) & bool_mask).sum().item()
                false_positive_confidence += ((pred_labels_confidence > 0) & (
                            test_image_ground_truth_batch == 0) & bool_mask).sum().item()

                for seg_idx in range(len(classes)):
                    seg_class = classes[seg_idx]
                    seg_ps = seg_idx + 1
                    true_positive_class[seg_class] += int(
                        torch.sum(
                            (pred_labels == seg_ps) & (test_image_ground_truth_batch == seg_ps) & bool_mask).item())
                    true_negative_class[seg_class] += int(
                        torch.sum(
                            (pred_labels != seg_ps) & (test_image_ground_truth_batch != seg_ps) & bool_mask).item())
                    false_positive_class[seg_class] += int(
                        torch.sum(
                            (pred_labels == seg_ps) & (test_image_ground_truth_batch != seg_ps) & bool_mask).item())
                    false_negative_class[seg_class] += int(
                        torch.sum(
                            (pred_labels != seg_ps) & (test_image_ground_truth_batch == seg_ps) & bool_mask).item())

                    true_positive_confidence_class[seg_class] += int(torch.sum((pred_labels_confidence == seg_ps) \
                                                                               & (
                                                                                           test_image_ground_truth_batch == seg_ps) & bool_mask).item())
                    true_negative_confidence_class[seg_class] += int(torch.sum((pred_labels_confidence != seg_ps) \
                                                                               & (
                                                                                           test_image_ground_truth_batch != seg_ps) & bool_mask).item())
                    false_positive_confidence_class[seg_class] += int(torch.sum((pred_labels_confidence == seg_ps) \
                                                                                & (
                                                                                            test_image_ground_truth_batch != seg_ps) & bool_mask).item())
                    false_negative_confidence_class[seg_class] += int(torch.sum((pred_labels_confidence != seg_ps) \
                                                                                & (
                                                                                            test_image_ground_truth_batch == seg_ps) & bool_mask).item())

                    true_positive_class_class[seg_class] += int(torch.sum((pred_labels_class == seg_ps) \
                                                                          & (
                                                                                      test_image_ground_truth_batch == seg_ps) & bool_mask).item())
                    true_negative_class_class[seg_class] += int(torch.sum((pred_labels_class != seg_ps) \
                                                                          & (
                                                                                      test_image_ground_truth_batch != seg_ps) & bool_mask).item())
                    false_positive_class_class[seg_class] += int(torch.sum((pred_labels_class == seg_ps) \
                                                                           & (
                                                                                       test_image_ground_truth_batch != seg_ps) & bool_mask).item())
                    false_negative_class_class[seg_class] += int(torch.sum((pred_labels_class != seg_ps) \
                                                                           & (
                                                                                       test_image_ground_truth_batch == seg_ps) & bool_mask).item())

                total_cum_loss += loss.item()

                tested += len(batch_indices)

                if test_only:
                    result_arr = result.detach().cpu().numpy()
                    for i in range(len(batch_indices)):
                        np.save(os.path.join(model_dir, 'result_{}.npy'.format(batch_indices[i])),
                                result_arr[i, :, :, :])

                del bool_mask, pred_labels, pred_labels_confidence, pred_labels_class
                del test_image_cat_batch, test_image_ground_truth_batch, test_image_ground_truth_mask_batch, \
                    test_image_ground_truth_deep[:], test_image_ground_truth_mask_deep[:]
                del result, deep_outputs[:]
                del test_image_ground_truth_deep, test_image_ground_truth_mask_deep, deep_outputs

                # gc.collect()
                # torch.cuda.empty_cache()

                pbar.update(len(batch_indices))

        total_loss_per_output /= len(validation_entries)
        total_cum_loss /= len(validation_entries)

        accuracy_per_output = (true_positive_per_output + true_negative_per_output).astype(np.float64) / (
                    true_positive_per_output + true_negative_per_output + false_positive_per_output + false_negative_per_output)
        precision_per_output = true_positive_per_output.astype(np.float64) / (
                    true_positive_per_output + false_positive_per_output)
        recall_per_output = true_positive_per_output.astype(np.float64) / (
                    true_positive_per_output + false_negative_per_output)

        accuracy_per_output = np.nan_to_num(accuracy_per_output, nan=0.0, posinf=0.0, neginf=0.0)
        precision_per_output = np.nan_to_num(precision_per_output, nan=0.0, posinf=0.0, neginf=0.0)
        recall_per_output = np.nan_to_num(recall_per_output, nan=0.0, posinf=0.0, neginf=0.0)

        for seg_class in classes:
            train_history["val_accuracy_" + seg_class].append(
                (true_positive_class[seg_class] + true_negative_class[seg_class]) / (
                            true_positive_class[seg_class] + true_negative_class[seg_class] + false_positive_class[
                        seg_class] + false_negative_class[seg_class]))
            if true_positive_class[seg_class] + false_positive_class[seg_class] == 0:
                train_history["val_precision_" + seg_class].append(0.0)
            else:
                train_history["val_precision_" + seg_class].append(
                    true_positive_class[seg_class] / (true_positive_class[seg_class] + false_positive_class[seg_class]))
            if true_positive_class[seg_class] + false_negative_class[seg_class] == 0:
                train_history["val_recall_" + seg_class].append(0.0)
            else:
                train_history["val_recall_" + seg_class].append(
                    true_positive_class[seg_class] / (true_positive_class[seg_class] + false_negative_class[seg_class]))

        train_history["val_loss_cum"].append(total_cum_loss)
        for k in range(pyr_height):
            train_history["val_loss{}".format(k)].append(total_loss_per_output[k])
            train_history["val_accuracy{}".format(k)].append(accuracy_per_output[k])
            train_history["val_precision{}".format(k)].append(precision_per_output[k])
            train_history["val_recall{}".format(k)].append(recall_per_output[k])

        confidence_accuracy = (true_positive_confidence + true_negative_confidence) / (
                true_positive_confidence + true_negative_confidence + false_positive_confidence + false_negative_confidence)
        if true_positive_confidence + false_positive_confidence == 0:
            confidence_precision = 0.0
        else:
            confidence_precision = true_positive_confidence / (
                    true_positive_confidence + false_positive_confidence)
        if true_positive_confidence + false_negative_confidence == 0:
            confidence_recall = 0.0
        else:
            confidence_recall = true_positive_confidence / (
                    true_positive_confidence + false_negative_confidence)

        train_history["val_confidence_accuracy"].append(confidence_accuracy)
        train_history["val_confidence_precision"].append(confidence_precision)
        train_history["val_confidence_recall"].append(confidence_recall)

        for seg_class in classes:
            train_history["val_confidence_accuracy_" + seg_class].append(
                (true_positive_confidence_class[seg_class] + true_negative_confidence_class[seg_class])
                / (true_positive_confidence_class[seg_class] + true_negative_confidence_class[seg_class] +
                   false_positive_confidence_class[seg_class] + false_negative_confidence_class[seg_class]))
            if true_positive_confidence_class[seg_class] + false_positive_confidence_class[seg_class] == 0:
                train_history["val_confidence_precision_" + seg_class].append(0.0)
            else:
                train_history["val_confidence_precision_" + seg_class].append(true_positive_confidence_class[seg_class]
                                                                              / (true_positive_confidence_class[
                                                                                     seg_class] +
                                                                                 false_positive_confidence_class[
                                                                                     seg_class]))
            if true_positive_confidence_class[seg_class] + false_negative_confidence_class[seg_class] == 0:
                train_history["val_confidence_recall_" + seg_class].append(0.0)
            else:
                train_history["val_confidence_recall_" + seg_class].append(true_positive_confidence_class[seg_class]
                                                                           / (true_positive_confidence_class[
                                                                                  seg_class] +
                                                                              false_negative_confidence_class[
                                                                                  seg_class]))

            train_history["val_class_accuracy_{}".format(seg_class)].append(
                (true_positive_class_class[seg_class] + true_negative_class_class[seg_class]) / (
                        true_positive_class_class[seg_class] + true_negative_class_class[seg_class] +
                        false_positive_class_class[seg_class] + false_negative_class_class[seg_class]))
            if true_positive_class_class[seg_class] + false_positive_class_class[seg_class] == 0:
                train_history["val_class_precision_{}".format(seg_class)].append(0.0)
            else:
                train_history["val_class_precision_{}".format(seg_class)].append(
                    true_positive_class_class[seg_class] / (
                            true_positive_class_class[seg_class] + false_positive_class_class[seg_class]))
            if true_positive_class_class[seg_class] + false_negative_class_class[seg_class] == 0:
                train_history["val_class_recall_{}".format(seg_class)].append(0.0)
            else:
                train_history["val_class_recall_{}".format(seg_class)].append(
                    true_positive_class_class[seg_class] / (
                            true_positive_class_class[seg_class] + false_negative_class_class[seg_class]))

    gc.collect()
    #torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a progressively supervised U-Net model with reconstructed WSI data.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use. Default 1.")
    parser.add_argument("--augmentation", action="store_true", help="Whether to use data augmentation. Default False.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate to use. Default 1e-5.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum to use. Default 0.9. This would be the momentum for SGD, and beta1 for Adam.")
    parser.add_argument("--second_momentum", type=float, default=0.999, help="Second momentum to use. Default 0.999. This would be beta2 for Adam. Ignored if SGD.")
    parser.add_argument("--gradient_clipping", type=float, default=0.0, help="Gradient clipping to use. Default 0.0.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use. Default 0.0.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch normalization. Default False.")
    parser.add_argument("--use_res_conv", action="store_true", help="Whether to use deeper residual convolutional networks. Default False.")
    parser.add_argument("--use_atrous_conv", action="store_true", help="Whether to use atrous convolutional networks. Default False.")
    parser.add_argument("--use_focal_loss", action="store_true", help="Whether to use focal loss. Default False.")
    parser.add_argument("--use_composite_focal_loss", action="store_true", help="Whether to use composite focal loss. Default False.")
    parser.add_argument("--use_separated_focal_loss", action="store_true", help="Whether to use separated focal loss. Default False.")
    parser.add_argument("--use_residual_atrous_conv", action="store_true", help="Whether to use residual atrous convolutions. Default False.")
    parser.add_argument("--use_heavy_boundary_in_separated_loss", action="store_true", help="Whether to use heavy boundary in separated focal loss. Default False. Must be used with --use_separated_focal_loss.")
    parser.add_argument("--use_softer_weights", action="store_true", help="Whether to use softer weights. Default False.")
    parser.add_argument("--use_amp", action="store_true", help="Whether to use automatic mixed precision. Default False.")
    parser.add_argument("--use_squeeze_excitation", action="store_true", help="Whether to use squeeze and excitation. Default False.")
    parser.add_argument("--use_initial_conv", action="store_true", help="Whether to use the initial 7x7 kernel convolution. Default False.")
    parser.add_argument("--use_suppressed_deepsupervision", action="store_true", help="Whether to use suppressed deep supervision. Default False.")
    parser.add_argument("--use_partially_suppressed_deepsupervision", action="store_true", help="Whether to use partially suppressed deep supervision. Default False.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels to use. Default 64.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[2, 3, 4, 6, 6, 7, 7], help="Number of hidden blocks for ResNets. Ignored if not resnet.")
    parser.add_argument("--bottleneck_expansion", type=int, default=1, help="The expansion factor of the bottleneck. Default 1.")
    parser.add_argument("--pyramid_height", type=int, default=4, help="Number of pyramid levels to use. Default 4.")
    parser.add_argument("--unet_attention", action="store_true", help="Whether to use attention in the U-Net. Default False.")
    parser.add_argument("--deep_exponent_base", type=float, default=2.0, help="The base of the exponent for the deep supervision loss. Default 2.0.")
    parser.add_argument("--mixup", type=float, default=0.0, help="The alpha value of mixup. Default 0, meaning no mixup.")
    parser.add_argument("--mixup_interlace", action="store_true", help="Whether to interlace the mixup. Default False.")
    parser.add_argument("--mixup_interlace_extras", type=int, default=1, help="The number of extra mixup interlacing. Default 1. Ignored if mixup_interlace=False.")
    parser.add_argument("--image_size", type=int, default=1024, help="The size of the images to use. Default 1024.")
    parser.add_argument("--image_stain_norm", action="store_true", help="Whether to stain normalize the images. Default False.")
    parser.add_argument("--use_async_sampling", type=int, default=0, help="Whether to use async sampling. Default 0, meaning no async sampling. The values represent the max buffer of the processes. If -1, the max buffer is unlimited.")
    parser.add_argument("--async_sampling_gpu", action="store_true", help="Whether to use GPU for async sampling. Default False. Ignored if use_async_sampling=0.")
    parser.add_argument("--num_extra_steps", type=int, default=0, help="Extra steps of gradient descent before the usual step in an epoch. Default 0.")
    parser.add_argument("--extra_training_subdata", type=str, default=None, help="Additional training subdata to mix with the main training data. Default None.")
    parser.add_argument("--extra_subdata_ratio", type=float, default=0.2, help="The ratio of the extra subdata to mix with the main training data. Default 0.2. If extra training subdata is not provided, this is ignored.")
    parser.add_argument("--mathematical", action="store_true", help="Whether to use mathematical loss. Default False.")
    parser.add_argument("--test_only", action="store_true", help="Whether to only test the model. Default False. If true, the epochs and number of extra steps are ignored, set to 1, 0.")

    model_data_manager.model_add_argparse_arguments(parser)

    args = parser.parse_args()

    model_dir, dataset_loader, training_entries, validation_entries, prev_model_checkpoint_dir, extra_info, train_subdata, val_subdata = model_data_manager.model_get_argparse_arguments(args, return_subdata_name=True)
    memory_logger = logging_memory_utils.obtain_memory_logger(model_dir, config.samplers_data_path == "")
    assert type(training_entries) == list
    assert type(validation_entries) == list

    extra_training_entries = None
    num_extra_trains = None
    total_training_len = len(training_entries)
    if args.extra_training_subdata is not None:
        extra_training_entries = model_data_manager.get_subdata_entry_list(args.extra_training_subdata)
        assert type(extra_training_entries) == list
        # convert to int lists
        training_int_id = np.unique(model_data_manager.get_intid_by_entry_index(training_entries))
        extra_training_int_id = np.unique(model_data_manager.get_intid_by_entry_index(extra_training_entries))
        valid_int_id = np.unique(model_data_manager.get_intid_by_entry_index(validation_entries))
        # check disjoint, and check contains
        intersection_empty = np.sum(np.searchsorted(extra_training_int_id, valid_int_id, side="left")
               < np.searchsorted(extra_training_int_id, valid_int_id, side="right")) == 0
        if not args.ignore_overlap:
            assert intersection_empty, "The validation set and the extra training set must be disjoint."
        del valid_int_id
        assert np.all(np.searchsorted(extra_training_int_id, training_int_id, side="left")
               < np.searchsorted(extra_training_int_id, training_int_id, side="right")), "The extra training set must contain all the training set."
        # get the diff
        diff = extra_training_int_id[np.searchsorted(training_int_id, extra_training_int_id, side="left") == np.searchsorted(training_int_id, extra_training_int_id, side="right")]
        del extra_training_entries[:], extra_training_entries, training_int_id, extra_training_int_id
        extra_training_entries = np.array(list(model_data_manager.get_entry_index_by_intid(diff)), dtype=object)
        del diff
        gc.collect()
        num_extra_trains = int(total_training_len * args.extra_subdata_ratio)
        total_training_len += num_extra_trains


    training_entries = np.array(training_entries, dtype=object)
    validation_entries = np.array(validation_entries, dtype=object)
    mixup = args.mixup
    mixup_interlace = args.mixup_interlace
    mixup_interlace_extras = args.mixup_interlace_extras
    image_size = args.image_size
    use_async_sampling = args.use_async_sampling
    num_extra_steps = args.num_extra_steps
    use_amp = args.use_amp
    blocks = args.hidden_blocks

    assert (not mixup_interlace) or mixup > 0.0, "Mixup interlace must be used with mixup."
    assert (not mixup_interlace) or args.use_async_sampling != 0, "Mixup interlace must be used with async sampling."
    assert type(mixup_interlace_extras) == int, "Mixup interlace extras must be an integer."
    assert mixup_interlace_extras >= 1, "Mixup interlace extras must be at least 1."

    num_epochs = args.epochs
    batch_size = args.batch_size
    augmentation = args.augmentation
    epochs_per_save = args.epochs_per_save
    image_pixels_round = 2 ** args.pyramid_height
    pyr_height = args.pyramid_height
    deep_exponent_base = args.deep_exponent_base
    use_focal_loss = args.use_focal_loss
    use_composite_focal_loss = args.use_composite_focal_loss
    use_separated_focal_loss = args.use_separated_focal_loss
    use_residual_atrous_conv = args.use_residual_atrous_conv
    use_suppressed_deepsupervision = args.use_suppressed_deepsupervision
    use_partially_suppressed_deepsupervision = args.use_partially_suppressed_deepsupervision
    use_softer_weights = args.use_softer_weights
    mathematical = args.mathematical
    test_only = args.test_only

    assert not (use_focal_loss and use_composite_focal_loss), "You cannot use both focal loss and composite focal loss."
    assert not (use_focal_loss and use_separated_focal_loss), "You cannot use both focal loss and separated focal loss."
    assert not (use_composite_focal_loss and use_separated_focal_loss), "You cannot use both composite focal loss and separated focal loss."
    assert (not use_separated_focal_loss) or args.use_atrous_conv, "You can only use separated focal loss with atrous convolutions."
    assert (not use_residual_atrous_conv) or use_separated_focal_loss, "You can only use residual atrous convolutions with separated focal loss."
    # assert not (use_suppressed_deepsupervision and use_partially_suppressed_deepsupervision), "You cannot use both suppressed and partially suppressed deep supervision."
    if test_only:
        num_epochs = 1
        num_extra_steps = 0


    assert type(blocks) == list, "Blocks must be a list."
    for k in blocks:
        assert type(k) == int, "Blocks must be a list of integers."
    
    print("Hidden channels: " + str(args.hidden_channels))
    print("Hidden blocks: " + str(blocks))

    if args.unet_attention:
        model = model_unet_attention.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                                    hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                                    use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height,
                                                    in_channels=4, use_atrous_conv=args.use_atrous_conv, atrous_outconv_split=use_separated_focal_loss, atrous_outconv_residual=use_residual_atrous_conv,
                                                    deep_supervision=True, squeeze_excitation=args.use_squeeze_excitation,
                                                    bottleneck_expansion=args.bottleneck_expansion,
                                                    res_conv_blocks=blocks, use_initial_conv=args.use_initial_conv).to(device=config.device)
    else:
        model = model_unet_base.UNetClassifier(num_classes=2, num_deep_multiclasses=args.pyramid_height - 1,
                                                hidden_channels=args.hidden_channels, use_batch_norm=args.use_batch_norm,
                                               use_res_conv=args.use_res_conv, pyr_height=args.pyramid_height, deep_supervision=True,
                                               in_channels=4, use_atrous_conv=args.use_atrous_conv, atrous_outconv_split=use_separated_focal_loss, atrous_outconv_residual=use_residual_atrous_conv,
                                               squeeze_excitation=args.use_squeeze_excitation, bottleneck_expansion=args.bottleneck_expansion,
                                               res_conv_blocks=blocks, use_initial_conv=args.use_initial_conv).to(device=config.device)

    single_training_step_compile = single_training_step#torch.compile(single_training_step)

    momentum = args.momentum
    second_momentum = args.second_momentum
    if args.test_only:
        optimizer = None
        print("---------------------------------- TESTING ONLY ----------------------------------")
    else:
        if args.optimizer.lower() == "adam":
            if args.weight_decay == 0.0:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(momentum, second_momentum))
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(momentum, second_momentum), weight_decay=args.weight_decay)
        elif args.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=momentum, weight_decay=args.weight_decay)
        else:
            print("Invalid optimizer. The available options are: adam, sgd.")
            exit(1)
        print("Using optimizer {} with learning rate {} and momentum {}.".format(args.optimizer, args.learning_rate, momentum))
        print("Using weight decay {}.".format(args.weight_decay))
        print("Using gradient clipping {}.".format(args.gradient_clipping))
        print("Using mixup {} and mixup_interlace {} with extras {}".format(mixup, mixup_interlace, mixup_interlace_extras))

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    if prev_model_checkpoint_dir is not None:
        model_checkpoint_path = os.path.join(prev_model_checkpoint_dir, "model.pt")
        optimizer_checkpoint_path = os.path.join(prev_model_checkpoint_dir, "optimizer.pt")

        model.load_state_dict(torch.load(model_checkpoint_path, map_location="cpu"))
        if not args.test_only:
            optimizer.load_state_dict(torch.load(optimizer_checkpoint_path, map_location="cpu"))

            for g in optimizer.param_groups:
                g["lr"] = args.learning_rate
                if args.weight_decay > 0.0:
                    assert "weight_decay" in g, "The optimizer does not have weight decay."
                    g["weight_decay"] = args.weight_decay
                if args.optimizer == "sgd":
                    g["momentum"] = momentum
                elif args.optimizer == "adam":
                    g["betas"] = (momentum, second_momentum)

        gc.collect()
        torch.cuda.empty_cache()

    model_config = {
        "model": "reconstructed_model_progressive_supervised_unet",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "augmentation": augmentation,
        "learning_rate": args.learning_rate,
        "momentum": momentum,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "epochs_per_save": epochs_per_save,
        "use_batch_norm": args.use_batch_norm,
        "use_res_conv": args.use_res_conv,
        "use_atrous_conv": args.use_atrous_conv,
        "use_focal_loss": args.use_focal_loss,
        "use_composite_focal_loss": args.use_composite_focal_loss,
        "use_separated_focal_loss": args.use_separated_focal_loss,
        "use_residual_atrous_conv": use_residual_atrous_conv,
        "use_heavy_boundary_in_separated_loss": args.use_heavy_boundary_in_separated_loss,
        "use_softer_weights": use_softer_weights,
        "use_amp": args.use_amp,
        "use_squeeze_excitation": args.use_squeeze_excitation,
        "use_initial_conv": args.use_initial_conv,
        "use_suppressed_deepsupervision": args.use_suppressed_deepsupervision,
        "use_partially_suppressed_deepsupervision": use_partially_suppressed_deepsupervision,
        "hidden_channels": args.hidden_channels,
        "hidden_blocks": args.hidden_blocks,
        "bottleneck_expansion": args.bottleneck_expansion,
        "pyramid_height": args.pyramid_height,
        "unet_attention": args.unet_attention,
        "deep_exponent_base": deep_exponent_base,
        "mixup": mixup,
        "mixup_interlace": mixup_interlace,
        "mixup_interlace_extras": mixup_interlace_extras,
        "image_size": args.image_size,
        "image_stain_norm": args.image_stain_norm,
        "use_async_sampling": use_async_sampling,
        "async_sampling_gpu": args.async_sampling_gpu,
        "num_extra_steps": num_extra_steps,
        "extra_training_subdata": args.extra_training_subdata,
        "extra_subdata_ratio": args.extra_subdata_ratio,
        "mathematical": mathematical,
        "training_script": "reconstructed_model_progressive_supervised_unet.py",
    }
    for key, value in extra_info.items():
        model_config[key] = value

    # Save the model config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

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

    train_history["confidence_accuracy"] = []
    train_history["confidence_precision"] = []
    train_history["confidence_recall"] = []
    train_history["val_confidence_accuracy"] = []
    train_history["val_confidence_precision"] = []
    train_history["val_confidence_recall"] = []

    print("Using class weights:")
    classes = ["blood_vessel", "boundary"]
    class_weights = [5.0, 5.0]
    if args.use_separated_focal_loss and args.use_heavy_boundary_in_separated_loss:
        if use_softer_weights:
            class_weights_composite = [1.5, 3.0]
        else:
            class_weights_composite = [1.0, 4.0]
    else:
        if use_softer_weights:
            class_weights_composite = [1.5, 7.5]
        else:
            class_weights_composite = [1.0, 10.0]
    num_classes = 2
    for k in range(len(classes)):
        seg_class = classes[k]
        train_history["accuracy_{}".format(seg_class)] = []
        train_history["val_accuracy_{}".format(seg_class)] = []
        train_history["precision_{}".format(seg_class)] = []
        train_history["val_precision_{}".format(seg_class)] = []
        train_history["recall_{}".format(seg_class)] = []
        train_history["val_recall_{}".format(seg_class)] = []

        train_history["confidence_accuracy_{}".format(seg_class)] = []
        train_history["val_confidence_accuracy_{}".format(seg_class)] = []
        train_history["confidence_precision_{}".format(seg_class)] = []
        train_history["val_confidence_precision_{}".format(seg_class)] = []
        train_history["confidence_recall_{}".format(seg_class)] = []
        train_history["val_confidence_recall_{}".format(seg_class)] = []

        train_history["class_accuracy_{}".format(seg_class)] = []
        train_history["val_class_accuracy_{}".format(seg_class)] = []
        train_history["class_precision_{}".format(seg_class)] = []
        train_history["val_class_precision_{}".format(seg_class)] = []
        train_history["class_recall_{}".format(seg_class)] = []
        train_history["val_class_recall_{}".format(seg_class)] = []
        print("Class {}: {}, {}".format(seg_class, class_weights[k], class_weights_composite[k]))

    if use_separated_focal_loss:
        class_weights_old = torch.tensor([1.0] + class_weights, dtype=torch.float32, device=config.device)
        class_weights = torch.tensor(class_weights[0], dtype=torch.float32, device=config.device)
    else:
        class_weights = torch.tensor([1.0] + class_weights, dtype=torch.float32, device=config.device)
    class_weights_composite = torch.tensor(class_weights_composite, dtype=torch.float32, device=config.device)

    # Initialize image sampler here
    if use_async_sampling == 0:
        print("Using synchronous image sampler")
        if args.extra_training_subdata is not None:
            train_sampler = image_wsi_sampling.get_image_sampler(args.extra_training_subdata, image_width=image_size, use_stainnet=args.image_stain_norm)
        else:
            train_sampler = image_wsi_sampling.get_image_sampler(train_subdata, image_width=image_size, use_stainnet=args.image_stain_norm)
        val_sampler = image_wsi_sampling.get_image_sampler(val_subdata, image_width=image_size, use_stainnet=args.image_stain_norm)
    else:
        print("Using asynchronous image sampler. GPU: {}".format(args.async_sampling_gpu))
        import image_wsi_sampling_async
        torch.multiprocessing.set_start_method("spawn")

        if args.extra_training_subdata is not None:
            train_sampler = image_wsi_sampling_async.get_image_sampler(args.extra_training_subdata, image_width=image_size, batch_size=batch_size, sampling_type="batch_random_image_mixup" if mixup > 0.0 else "batch_random_image",
                                                                deep_supervision_outputs=pyr_height - 1, buffer_max_size=use_async_sampling, use_stainnet=args.image_stain_norm, use_gpu=args.async_sampling_gpu)
        else:
            train_sampler = image_wsi_sampling_async.get_image_sampler(train_subdata, image_width=image_size, batch_size=batch_size, sampling_type="batch_random_image_mixup" if mixup > 0.0 else "batch_random_image",
                                                                deep_supervision_outputs=pyr_height - 1, buffer_max_size=use_async_sampling, use_stainnet=args.image_stain_norm, use_gpu=args.async_sampling_gpu)
        if mixup_interlace:
            train_sampler_interlace_nomixup = image_wsi_sampling_async.get_image_sampler(train_subdata, image_width=image_size, batch_size=batch_size, sampling_type="batch_random_image",
                                                                deep_supervision_outputs=pyr_height - 1, buffer_max_size=use_async_sampling, use_stainnet=args.image_stain_norm, use_gpu=args.async_sampling_gpu)
        val_sampler = image_wsi_sampling_async.get_image_sampler(val_subdata, image_width=image_size, batch_size=batch_size, sampling_type="batch_random_image", deep_supervision_outputs=pyr_height - 1,
                                                                 buffer_max_size=use_async_sampling, use_stainnet=args.image_stain_norm, use_gpu=args.async_sampling_gpu)

        def async_request_images_train():
            # request samples from the sampler
            if mixup > 0.0:
                training_entries_shuffle = rng.permutation(training_entries)
                training_entries_shuffle2 = rng.permutation(training_entries)
                if extra_training_entries is not None:
                    extra1_size = num_extra_trains // 2
                    extra2_size = num_extra_trains - extra1_size
                    training_entries_shuffle = np.concatenate((training_entries_shuffle,
                                                               rng.choice(extra_training_entries, size=num_extra_trains, replace=False)))
                    training_entries_shuffle2 = np.concatenate((training_entries_shuffle2,
                                                                rng.choice(training_entries, size=extra1_size, replace=False)))
                    training_entries_shuffle2 = np.concatenate((training_entries_shuffle2,
                                                                rng.choice(extra_training_entries, size=extra2_size, replace=False)))

                    perm = rng.permutation(np.arange(len(training_entries_shuffle)))
                    training_entries_shuffle = training_entries_shuffle[perm]
                    training_entries_shuffle2 = training_entries_shuffle2[perm]
                    del perm

                # add to sampler
                trained = 0
                while trained < total_training_len:
                    batch_end = min(trained + batch_size, total_training_len)
                    batch_indices = training_entries_shuffle[trained:batch_end]
                    batch_indices2 = training_entries_shuffle2[trained:batch_end]
                    train_sampler.request_load_sample_mixup(list(batch_indices), list(batch_indices2), mixup, augmentation=augmentation)

                    trained += len(batch_indices)
            else:
                training_entries_shuffle = rng.permutation(training_entries)
                if extra_training_entries is not None:
                    training_entries_shuffle = np.concatenate((training_entries_shuffle,
                                                                rng.choice(extra_training_entries, size=num_extra_trains, replace=False)))
                    training_entries_shuffle = rng.permutation(training_entries_shuffle)

                # add to sampler
                trained = 0
                while trained < total_training_len:
                    batch_end = min(trained + batch_size, total_training_len)
                    batch_indices = training_entries_shuffle[trained:batch_end]
                    train_sampler.request_load_sample(list(batch_indices), augmentation=augmentation, random_location=True)

                    trained += len(batch_indices)

            # if interlace, request samples from the nomixup sampler
            if mixup_interlace:
                for k in range(mixup_interlace_extras):
                    # shuffle
                    training_entries_shuffle = rng.permutation(training_entries)
                    # add to sampler
                    trained = 0
                    while trained < len(training_entries):
                        batch_end = min(trained + batch_size, len(training_entries))
                        batch_indices = training_entries_shuffle[trained:batch_end]
                        train_sampler_interlace_nomixup.request_load_sample(list(batch_indices), augmentation=augmentation,
                                                          random_location=True)

                        trained += len(batch_indices)

        def async_request_images_val():
            tested = 0
            while tested < len(validation_entries):
                batch_end = min(tested + batch_size, len(validation_entries))
                batch_indices = validation_entries[tested:batch_end]

                val_sampler.request_load_sample(list(batch_indices), augmentation=False, random_location=False)

                tested += len(batch_indices)

    memory_logger.log("CUDA memory before training:")
    try:
        if use_async_sampling != 0:
            async_request_images_train() # preload the training samples

        # Training loop
        for epoch in range(num_epochs):
            ctime = time.time()
            # Train the model
            # ----------------------------------------- Run extra steps -----------------------------------------
            print()
            print("Training {} extra steps...".format(num_extra_steps))
            print()
            for extra_step in range(num_extra_steps):
                training_step(None)
                memory_logger.log("CUDA memory after extra step {} in epoch {}".format(extra_step, epoch))

            # ----------------------------------------- Main training step -----------------------------------------
            print()
            print("Training.....")
            print()
            # if test only, the metrics are all zero
            if test_only:
                train_history["loss_cum"].append(0.0)
                for k in range(pyr_height):
                    train_history["loss{}".format(k)].append(0.0)
                    train_history["accuracy{}".format(k)].append(0.0)
                    train_history["precision{}".format(k)].append(0.0)
                    train_history["recall{}".format(k)].append(0.0)

                train_history["confidence_accuracy"].append(0.0)
                train_history["confidence_precision"].append(0.0)
                train_history["confidence_recall"].append(0.0)

                for k in range(len(classes)):
                    seg_class = classes[k]
                    train_history["accuracy_{}".format(seg_class)].append(0.0)
                    train_history["precision_{}".format(seg_class)].append(0.0)
                    train_history["recall_{}".format(seg_class)].append(0.0)

                    train_history["confidence_accuracy_{}".format(seg_class)].append(0.0)
                    train_history["confidence_precision_{}".format(seg_class)].append(0.0)
                    train_history["confidence_recall_{}".format(seg_class)].append(0.0)

                    train_history["class_accuracy_{}".format(seg_class)].append(0.0)
                    train_history["class_precision_{}".format(seg_class)].append(0.0)
                    train_history["class_recall_{}".format(seg_class)].append(0.0)
            else:
                training_step(train_history)
            memory_logger.log("CUDA memory after training step in epoch {}".format(epoch))

            if use_async_sampling != 0:
                async_request_images_train() # request preloading training images for next epoch

            # ----------------------------------------- Test the model -----------------------------------------
            validation_step(train_history)

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

            print("{} (Confidence Accuracy)".format(train_history["confidence_accuracy"][-1]))
            print("{} (Val Confidence Accuracy)".format(train_history["val_confidence_accuracy"][-1]))
            print("{} (Confidence Precision)".format(train_history["confidence_precision"][-1]))
            print("{} (Val Confidence Precision)".format(train_history["val_confidence_precision"][-1]))
            print("{} (Confidence Recall)".format(train_history["confidence_recall"][-1]))
            print("{} (Val Confidence Recall)".format(train_history["val_confidence_recall"][-1]))
            for seg_class in classes:
                print("{} (Confidence Accuracy {})".format(train_history["confidence_accuracy_" + seg_class][-1], seg_class))
                print("{} (Val Confidence Accuracy {})".format(train_history["val_confidence_accuracy_" + seg_class][-1], seg_class))
                print("{} (Confidence Precision {})".format(train_history["confidence_precision_" + seg_class][-1], seg_class))
                print("{} (Val Confidence Precision {})".format(train_history["val_confidence_precision_" + seg_class][-1], seg_class))
                print("{} (Confidence Recall {})".format(train_history["confidence_recall_" + seg_class][-1], seg_class))
                print("{} (Val Confidence Recall {})".format(train_history["val_confidence_recall_" + seg_class][-1], seg_class))

            for seg_class in classes:
                print("{} (Class Accuracy {})".format(train_history["class_accuracy_" + seg_class][-1], seg_class))
                print("{} (Val Class Accuracy {})".format(train_history["val_class_accuracy_" + seg_class][-1], seg_class))
                print("{} (Class Precision {})".format(train_history["class_precision_" + seg_class][-1], seg_class))
                print("{} (Val Class Precision {})".format(train_history["val_class_precision_" + seg_class][-1], seg_class))
                print("{} (Class Recall {})".format(train_history["class_recall_" + seg_class][-1], seg_class))
                print("{} (Val Class Recall {})".format(train_history["val_class_recall_" + seg_class][-1], seg_class))

            print("Learning Rate: {}".format(args.learning_rate))
            print("")

            # Save the training history
            train_history_save = pd.DataFrame(train_history)
            train_history_save.to_csv(os.path.join(model_dir, "train_history.csv"), index=False)
            del train_history_save

            ctime = time.time()

            gc.collect()
            #torch.cuda.empty_cache()

            memory_logger.log("CUDA memory after validation in epoch {}".format(epoch))

            # Save the model and optimizer
            if epoch % epochs_per_save == 0 and epoch > 0 and not test_only:
                torch.save(model.state_dict(), os.path.join(model_dir, "model_epoch{}.pt".format(epoch)))
                torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_epoch{}.pt".format(epoch)))

                memory_logger.log("CUDA memory after saving in epoch {}".format(epoch))

        print("Training Complete")

        # Save the model and optimizer
        if not test_only:
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pt"))
        # Save the training history by converting it to a dataframe
        train_history = pd.DataFrame(train_history)
        train_history.to_csv(os.path.join(model_dir, "train_history.csv"), index=False)

        # Plot the training history
        fig, axes = plt.subplots(2 + 2 * pyr_height + 3 * num_classes, 1, figsize=(12, 8 + 8 * pyr_height + 12 * num_classes))
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

        # plot the confidence predictions
        axes[1 + 2 * pyr_height + num_classes].plot(train_history["confidence_accuracy"], label="Train Accuracy")
        axes[1 + 2 * pyr_height + num_classes].plot(train_history["val_confidence_accuracy"], label="Validation Accuracy")
        axes[1 + 2 * pyr_height + num_classes].plot(train_history["confidence_precision"], label="Train Precision")
        axes[1 + 2 * pyr_height + num_classes].plot(train_history["val_confidence_precision"], label="Validation Precision")
        axes[1 + 2 * pyr_height + num_classes].plot(train_history["confidence_recall"], label="Train Recall")
        axes[1 + 2 * pyr_height + num_classes].plot(train_history["val_confidence_recall"], label="Validation Recall")
        axes[1 + 2 * pyr_height + num_classes].set_title("Accuracy, Precision, and Recall (Confidence)")
        axes[1 + 2 * pyr_height + num_classes].set_xlabel("Epoch")
        axes[1 + 2 * pyr_height + num_classes].set_ylabel("Metric")
        axes[1 + 2 * pyr_height + num_classes].legend()

        for k in range(num_classes):
            axes[2 + 2 * pyr_height + num_classes + k].plot(train_history["confidence_accuracy_" + classes[k]], label="Train Accuracy")
            axes[2 + 2 * pyr_height + num_classes + k].plot(train_history["val_confidence_accuracy_" + classes[k]], label="Validation Accuracy")
            axes[2 + 2 * pyr_height + num_classes + k].plot(train_history["confidence_precision_" + classes[k]], label="Train Precision")
            axes[2 + 2 * pyr_height + num_classes + k].plot(train_history["val_confidence_precision_" + classes[k]], label="Validation Precision")
            axes[2 + 2 * pyr_height + num_classes + k].plot(train_history["confidence_recall_" + classes[k]], label="Train Recall")
            axes[2 + 2 * pyr_height + num_classes + k].plot(train_history["val_confidence_recall_" + classes[k]], label="Validation Recall")
            axes[2 + 2 * pyr_height + num_classes + k].set_title("Accuracy, Precision, and Recall (Confidence {})".format(classes[k]))
            axes[2 + 2 * pyr_height + num_classes + k].set_xlabel("Epoch")
            axes[2 + 2 * pyr_height + num_classes + k].set_ylabel("Metric")
            axes[2 + 2 * pyr_height + num_classes + k].legend()

        for k in range(num_classes):
            axes[2 + 2 * pyr_height + 2 * num_classes + k].plot(train_history["class_accuracy_" + classes[k]], label="Train Accuracy")
            axes[2 + 2 * pyr_height + 2 * num_classes + k].plot(train_history["val_class_accuracy_" + classes[k]], label="Validation Accuracy")
            axes[2 + 2 * pyr_height + 2 * num_classes + k].plot(train_history["class_precision_" + classes[k]], label="Train Precision")
            axes[2 + 2 * pyr_height + 2 * num_classes + k].plot(train_history["val_class_precision_" + classes[k]], label="Validation Precision")
            axes[2 + 2 * pyr_height + 2 * num_classes + k].plot(train_history["class_recall_" + classes[k]], label="Train Recall")
            axes[2 + 2 * pyr_height + 2 * num_classes + k].plot(train_history["val_class_recall_" + classes[k]], label="Validation Recall")
            axes[2 + 2 * pyr_height + 2 * num_classes + k].set_title("Accuracy, Precision, and Recall (Class {})".format(classes[k]))
            axes[2 + 2 * pyr_height + 2 * num_classes + k].set_xlabel("Epoch")
            axes[2 + 2 * pyr_height + 2 * num_classes + k].set_ylabel("Metric")
            axes[2 + 2 * pyr_height + 2 * num_classes + k].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "train_history.png"))

        if use_async_sampling != 0:
            train_sampler.terminate()
            val_sampler.terminate()
            if mixup_interlace:
                train_sampler_interlace_nomixup.terminate()
    except KeyboardInterrupt:
        print("Training Interrupted. Saving Model...")
        # Save the model and optimizer
        if not test_only:
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pt"))

        if use_async_sampling != 0:
            train_sampler.terminate()
            val_sampler.terminate()
            if mixup_interlace:
                train_sampler_interlace_nomixup.terminate()
    except Exception as e:
        traceback.print_exc()
        print("Training failed! Terminating...")
        if use_async_sampling != 0:
            train_sampler.terminate()
            val_sampler.terminate()
            if mixup_interlace:
                train_sampler_interlace_nomixup.terminate()

    memory_logger.close()
