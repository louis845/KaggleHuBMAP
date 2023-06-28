import obtain_reconstructed_binary_segmentation
import inference_reconstructed_base
import image_wsi_sampling
import torch
import model_data_manager
import matplotlib.pyplot as plt

combined_sampler = image_wsi_sampling.get_image_sampler("dataset1_regional_split2", image_width=768)
loader = model_data_manager.get_dataset_dataloader(None)


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
    combined_torch2, gt, gt_mask = combined_sampler.obtain_random_image_from_tile(tile_id, augmentation=False,
                                                                                  random_location=False)
    blood_vessel_gt = loader.get_segmentation_mask(tile_id, "blood_vessel")

    assert combined_torch.shape == combined_torch2.shape
    assert torch.allclose(combined_torch, combined_torch2)
def display_tile(tile_id):
    combined_torch = inference_reconstructed_base.load_combined(tile_id)
    combined_torch2, gt, gt_mask = combined_sampler.obtain_random_image_from_tile(tile_id, augmentation=False,
                                                                                  random_location=False)
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
    for tile_id in model_data_manager.get_subdata_entry_list("dataset1_regional_split2"):
        check_tile_id(tile_id)