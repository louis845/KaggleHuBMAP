import gc

import numpy as np
import torch
import torchvision.transforms.functional
import cv2

import model_data_manager
import config

def apply_random_shear(displacement_field, xory="x", image_size=512, image_pad=1534, magnitude_low=10000.0, magnitude_high=16000.0):
    diff = (image_pad - image_size) // 2
    x = np.random.randint(low=0, high=image_size) + diff
    y = np.random.randint(low=0, high=image_size) + diff
    sigma = np.random.uniform(low=100.0, high=200.0)
    magnitude = np.random.uniform(low=magnitude_low, high=magnitude_high) * np.random.choice([-1, 1])

    width = image_size

    expand_left = min(x, width)
    expand_right = min(image_size - x, width + 1)
    expand_top = min(y, width)
    expand_bottom = min(image_size - y, width + 1)

    if xory == "x":
        displacement_field[0, x - expand_left:x + expand_right, y - expand_top:y + expand_bottom, 0:1] += \
            (np.expand_dims(cv2.getGaussianKernel(ksize=width * 2 + 1, sigma=sigma), axis=-1) * cv2.getGaussianKernel(
                ksize=width * 2 + 1, sigma=sigma) * magnitude)[width - expand_left:width + expand_right,
            width - expand_top:width + expand_bottom, :]
    else:
        displacement_field[0, x - expand_left:x + expand_right, y - expand_top:y + expand_bottom, 1:2] += \
            (np.expand_dims(cv2.getGaussianKernel(ksize=width * 2 + 1, sigma=sigma),
                            axis=-1) * cv2.getGaussianKernel(
                ksize=width * 2 + 1, sigma=sigma) * magnitude)[width - expand_left:width + expand_right,
            width - expand_top:width + expand_bottom, :]

def generate_displacement_field(image_size=512, image_pad=1534, dtype=torch.float32, device=config.device):
    displacement_field = np.zeros(shape=(1, image_pad, image_pad, 2), dtype=np.float32)

    type = np.random.choice(5)
    if type == 0:
        magnitude_low = 0.0
        magnitude_high = 1000.0
    elif type == 1:
        magnitude_low = 1000.0
        magnitude_high = 4000.0
    elif type == 2:
        magnitude_low = 4000.0
        magnitude_high = 7000.0
    elif type == 3:
        magnitude_low = 7000.0
        magnitude_high = 10000.0
    else:
        magnitude_low = 10000.0
        magnitude_high = 16000.0

    for k in range(4):
        apply_random_shear(displacement_field, xory="x", image_size=image_size, image_pad=image_pad, magnitude_low=magnitude_low, magnitude_high=magnitude_high)
        apply_random_shear(displacement_field, xory="y", image_size=image_size, image_pad=image_pad, magnitude_low=magnitude_low, magnitude_high=magnitude_high)

    displacement_field = torch.tensor(displacement_field, dtype=dtype, device=device)

    return displacement_field


def sample_images(indices: np.ndarray, dataset_loader: model_data_manager.DatasetDataLoader, ground_truth_mask="blood_vessel", rotation_augmentation=False, multiclass_labels_dict=None, deep_supervision_downsamples=0,
                  in_channels=3, image_height=512, image_width=512, crop_height=512, crop_width=512):
    """
    Sample images from the dataset.
    :param indices: The indices of the samples
    :param dataset_loader: The (images) loader of the dataset
    :param ground_truth_mask: Which ground truth mask to use in the dataset
    :param rotation_augmentation: Whether to do rotation augmentation and elastic deformation augmentation
    :param multiclass_labels_dict: The dictionary of the multiclass labels. If None, then the multiclass labels are not used, and returned image_multiclass_gt_batch, image_multiclass_gt_ds_batch are None.
    :param deep_supervision_downsamples: How many steps of stride2 maxpool downsampling are performed for deep supervision.
    :param in_channels: Number of input channels for the images
    :param image_height: The original height of the images
    :param image_width: The original width of the images
    :param crop_height: The height of the randomly cropped images
    :param crop_width: The width of the randomly cropped images
    :return:    image_data_batch: The batch of images
                image_ground_truth_batch: The batch of ground truth masks
                image_multiclass_gt_batch: The batch of multiclass ground truth masks
                image_ground_truth_ds_batch: The batch of ground truth masks for deep supervision
                image_multiclass_gt_ds_batch: The batch of multiclass ground truth masks for deep supervision

    """
    with torch.no_grad():
        length = len(indices)
        use_multiclass = multiclass_labels_dict is not None

        image_data_batch = torch.zeros((length, in_channels, image_height, image_width), dtype=torch.float32, device=config.device)
        image_ground_truth_batch = torch.zeros((length, image_height, image_width), dtype=torch.float32, device=config.device)
        if use_multiclass:
            image_multiclass_gt_batch = torch.zeros((length, image_height, image_width), dtype=torch.long, device=config.device)
        else:
            image_multiclass_gt_batch = None

        for k in range(length):
            image_data_batch[k, :, :, :] = torch.tensor(dataset_loader.get_image_data(indices[k]), dtype=torch.float32, device=config.device).permute(2, 0, 1)
            seg_mask = dataset_loader.get_segmentation_mask(indices[k], ground_truth_mask)
            image_ground_truth_batch[k, :, :] = torch.tensor(seg_mask, dtype=torch.float32, device=config.device)

            if use_multiclass:
                class_labels = multiclass_labels_dict[indices[k]]
                image_multiclass_gt_batch[k, :, :] = torch.tensor(class_labels, dtype=torch.long, device=config.device)

        # rotation / elastic deformation augmentation
        if rotation_augmentation:
            # flip the images
            if np.random.uniform(0, 1) < 0.5:
                image_data_batch = torch.flip(image_data_batch, dims=[3])
                image_ground_truth_batch = torch.flip(image_ground_truth_batch, dims=[2])
                if use_multiclass:
                    image_multiclass_gt_batch = torch.flip(image_multiclass_gt_batch, dims=[2])

            # apply elastic deformation
            image_data_batch = torch.nn.functional.pad(image_data_batch, (image_height - 1, image_height - 1, image_width - 1, image_width - 1), mode="reflect")
            image_ground_truth_batch = torch.nn.functional.pad(image_ground_truth_batch, (image_height - 1, image_height - 1, image_width - 1, image_width - 1), mode="reflect")
            if use_multiclass:
                image_multiclass_gt_batch = torch.nn.functional.pad(
                    image_multiclass_gt_batch.to(torch.float32),
                    (image_height - 1, image_height - 1, image_width - 1, image_width - 1), mode="reflect").to(torch.long)

            displacement_field = generate_displacement_field()
            image_data_batch = torchvision.transforms.functional.elastic_transform(image_data_batch, displacement_field)
            image_ground_truth_batch = torchvision.transforms.functional.elastic_transform(
                image_ground_truth_batch.unsqueeze(1), displacement_field).squeeze(1)
            if use_multiclass:
                image_multiclass_gt_batch = torchvision.transforms.functional.elastic_transform(
                    image_multiclass_gt_batch.unsqueeze(1), displacement_field,
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST).squeeze(1)

            # apply rotation
            angle_in_deg = np.random.uniform(0, 360)
            image_data_batch = torchvision.transforms.functional.rotate(image_data_batch, angle_in_deg)
            image_ground_truth_batch = torchvision.transforms.functional.rotate(image_ground_truth_batch, angle_in_deg)
            if use_multiclass:
                image_multiclass_gt_batch = torchvision.transforms.functional.rotate(image_multiclass_gt_batch,
                                                angle_in_deg, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

            # crop to original size in center
            image_data_batch = image_data_batch[..., image_height - 1:-image_height + 1, image_width - 1:-image_width + 1]
            image_ground_truth_batch = image_ground_truth_batch[..., image_height - 1:-image_height + 1, image_width - 1:-image_width + 1]
            if use_multiclass:
                image_multiclass_gt_batch = image_multiclass_gt_batch[..., image_height - 1:-image_height + 1, image_width - 1:-image_width + 1]

        # do random cropping here
        if crop_width < image_width or crop_height < image_height:
            crop_y = np.random.randint(0, image_height - crop_height + 1)
            crop_x = np.random.randint(0, image_width - crop_width + 1)

            image_data_batch = image_data_batch[..., crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
            image_ground_truth_batch = image_ground_truth_batch[..., crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
            if use_multiclass:
                image_multiclass_gt_batch = image_multiclass_gt_batch[..., crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

        # do deep supervision downsampling here
        image_ground_truth_ds_batch = []
        if use_multiclass:
            image_multiclass_gt_ds_batch = []
        else:
            image_multiclass_gt_ds_batch = None
        image_ground_truth_pooled_batch = image_ground_truth_batch.view(length, 1, crop_height, crop_width)
        for k in range(deep_supervision_downsamples):
            scale_factor = 2 ** (k + 1)
            image_ground_truth_pooled_batch = torch.nn.functional.max_pool2d(image_ground_truth_pooled_batch, kernel_size=2, stride=2)
            image_ground_truth_ds_batch.append(image_ground_truth_pooled_batch.view(length, crop_height // scale_factor, crop_width // scale_factor))
            if use_multiclass:
                image_multiclass_gt_pooled = torch.nn.functional.max_pool2d(image_multiclass_gt_batch.view(length, 1, crop_height, crop_width)
                    .to(torch.float32), kernel_size=scale_factor, stride=scale_factor).to(torch.long).squeeze(1)
                image_multiclass_gt_ds_batch.append(image_multiclass_gt_pooled)

    return image_data_batch, image_ground_truth_batch, image_multiclass_gt_batch, image_ground_truth_ds_batch, image_multiclass_gt_ds_batch

def sample_images_mixup(indices1: np.ndarray, indices2: np.ndarray, dataset_loader: model_data_manager.DatasetDataLoader,
                        mixup_alpha=0.2, ground_truth_mask="blood_vessel", rotation_augmentation=False,
                        multiclass_labels_dict=None, num_classes=0, deep_supervision_downsamples=0,
                        in_channels=3, image_height=512, image_width=512, crop_height=512, crop_width=512):
    """
    Samples images from the dataset and applies mixup augmentation. Mixup augmentation is described in the paper
    "mixup: Beyond Empirical Risk Minimization" by Hongyi Zhang et al. (https://arxiv.org/abs/1710.09412).
    The rest of the params and the return values are the same as sample_images.
    :param indices1: Indices to use as mixup samples 1.
    :param indices2: Indices to use as mixup samples 2.
    :param mixup_alpha: Alpha parameter for mixup augmentation.
    """
    assert len(indices1) == len(indices2), "indices1 and indices2 must have the same length"
    assert multiclass_labels_dict is None or num_classes > 0, "num_classes must be provided if multiclass_labels_dict is not None"
    use_multiclass = multiclass_labels_dict is not None

    with torch.no_grad():
        image_data_batch1, image_ground_truth_batch1, image_multiclass_gt_batch1, image_ground_truth_ds_batch1,\
            image_multiclass_gt_ds_batch1 = sample_images(indices1, dataset_loader, ground_truth_mask, rotation_augmentation,
                                                          multiclass_labels_dict, deep_supervision_downsamples,
                                                            in_channels, image_height, image_width, crop_height, crop_width)

        image_data_batch2, image_ground_truth_batch2, image_multiclass_gt_batch2, image_ground_truth_ds_batch2, \
            image_multiclass_gt_ds_batch2 = sample_images(indices2, dataset_loader, ground_truth_mask,
                                                          rotation_augmentation,
                                                          multiclass_labels_dict, deep_supervision_downsamples,
                                                          in_channels, image_height, image_width, crop_height, crop_width)

        gc.collect()
        torch.cuda.empty_cache()

        # apply mixup augmentation
        lambda_interp = np.random.beta(mixup_alpha, mixup_alpha)
        image_data_batch = lambda_interp * image_data_batch1 + (1 - lambda_interp) * image_data_batch2
        image_ground_truth_batch = lambda_interp * image_ground_truth_batch1 + (1 - lambda_interp) * image_ground_truth_batch2
        if use_multiclass:
            image_multiclass_gt_batch = lambda_interp * torch.nn.functional.one_hot(image_multiclass_gt_batch1, num_classes=num_classes+1).to(dtype=torch.float32).permute(0, 3, 1, 2)\
                                            + (1 - lambda_interp) * torch.nn.functional.one_hot(image_multiclass_gt_batch2, num_classes=num_classes+1).to(dtype=torch.float32).permute(0, 3, 1, 2)

        del image_data_batch1, image_data_batch2, image_ground_truth_batch1, image_ground_truth_batch2
        if use_multiclass:
            del image_multiclass_gt_batch1, image_multiclass_gt_batch2

        gc.collect()
        torch.cuda.empty_cache()

        image_ground_truth_ds_batch = []
        if use_multiclass:
            image_multiclass_gt_ds_batch = []
        else:
            image_multiclass_gt_ds_batch = None
        for k in range(deep_supervision_downsamples):
            image_ground_truth_ds_batch.append(lambda_interp * image_ground_truth_ds_batch1[k] + (1 - lambda_interp) * image_ground_truth_ds_batch2[k])
            if use_multiclass:
                image_multiclass_gt_ds_batch.append(lambda_interp * torch.nn.functional.one_hot(image_multiclass_gt_ds_batch1[k], num_classes=num_classes+1).to(dtype=torch.float32).permute(0, 3, 1, 2)
                                                    + (1 - lambda_interp) * torch.nn.functional.one_hot(image_multiclass_gt_ds_batch2[k], num_classes=num_classes+1).to(dtype=torch.float32).permute(0, 3, 1, 2))

        del image_ground_truth_ds_batch1[:], image_ground_truth_ds_batch2[:]
        del image_ground_truth_ds_batch1, image_ground_truth_ds_batch2
        if use_multiclass:
            del image_multiclass_gt_ds_batch1[:], image_multiclass_gt_ds_batch2[:]
            del image_multiclass_gt_ds_batch1, image_multiclass_gt_ds_batch2

        gc.collect()
        torch.cuda.empty_cache()

        return image_data_batch, image_ground_truth_batch, image_multiclass_gt_batch, image_ground_truth_ds_batch, image_multiclass_gt_ds_batch