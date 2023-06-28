import obtain_reconstructed_binary_segmentation
import inference_reconstructed_base
import image_wsi_sampling
import image_wsi_sampling_async
import torch
import model_data_manager
import matplotlib.pyplot as plt
import config
import tqdm

def show_tensor(tensor):
    """
    Displays a 2D torch tensor (grayscale) using matplotlib.
    """
    plt.imshow(tensor.cpu().numpy(), cmap='gray')
    plt.show()


def show_tensor_RGB(tensor):
    """
    Displays a RGB image represented by a (3, H, W) torch tensor using matplotlib.
    """
    # Permute the dimensions to (H, W, 3) and convert to numpy array
    tensor_np = tensor.permute(1, 2, 0).to(torch.uint8).cpu().numpy()

    # Display the image using matplotlib
    plt.imshow(tensor_np)
    plt.show()

def check_tile_id(tile_id):
    combined_torch = inference_reconstructed_base.load_combined(tile_id)
    # if type of combined sampler is MultipleImageSampler
    if isinstance(combined_sampler, image_wsi_sampling.MultipleImageSampler):
        combined_torch2, gt, gt_mask = combined_sampler.obtain_random_image_from_tile(tile_id, augmentation=False,
                                                                                      random_location=False)
    elif isinstance(combined_sampler, image_wsi_sampling_async.MultipleImageSamplerAsync):
        combined_torch2, gt, gt_mask = combined_sampler.get_samples(device=config.device, length=1)
        combined_torch2 = combined_torch2[0, ...]
        gt = gt[0, ...]
        gt_mask = gt_mask[0, ...]
    else:
        raise ValueError("Unknown type of combined sampler")

    blood_vessel_gt = loader.get_segmentation_mask(tile_id, "blood_vessel")

    assert combined_torch.shape == combined_torch2.shape
    assert torch.allclose(combined_torch, combined_torch2)
def display_tile(tile_id):
    combined_torch = inference_reconstructed_base.load_combined(tile_id)
    if isinstance(combined_sampler, image_wsi_sampling.MultipleImageSampler):
        combined_torch2, gt, gt_mask = combined_sampler.obtain_random_image_from_tile(tile_id, augmentation=False,
                                                                                      random_location=False)
    elif isinstance(combined_sampler, image_wsi_sampling_async.MultipleImageSamplerAsync):
        combined_torch2, gt, gt_mask = combined_sampler.get_samples(device=config.device, length=1)
        combined_torch2 = combined_torch2[0, ...]
        gt = gt[0, ...]
        gt_mask = gt_mask[0, ...]
    else:
        raise ValueError("Unknown type of combined sampler")
    blood_vessel_gt = loader.get_segmentation_mask(tile_id, "blood_vessel")

    show_tensor(combined_torch[3, ...] * 255)
    show_tensor(combined_torch2[3, ...] * 255)

    show_tensor_RGB(combined_torch[:3, ...])
    show_tensor_RGB(combined_torch2[:3, ...])

    show_tensor(gt * 128)
    show_tensor(gt_mask * 255)

    plt.imshow(blood_vessel_gt)
    plt.show()

if __name__ == "__main__":
    dataset = "dataset1_regional_split1"

    #combined_sampler = image_wsi_sampling.get_image_sampler(dataset, image_width=768)
    combined_sampler = image_wsi_sampling_async.get_image_sampler(dataset, image_width=768, sampling_type="batch_random_image", buffer_max_size=100)
    for tile_id in model_data_manager.get_subdata_entry_list(dataset):
        combined_sampler.request_load_sample([tile_id], augmentation=False, random_location=False)

    loader = model_data_manager.get_dataset_dataloader(None)
    for tile_id in tqdm.tqdm(model_data_manager.get_subdata_entry_list(dataset)):
        check_tile_id(tile_id)

    combined_sampler.terminate()