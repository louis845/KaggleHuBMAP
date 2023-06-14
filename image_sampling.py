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