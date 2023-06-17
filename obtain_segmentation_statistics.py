import numpy as np
import model_data_manager

loader = model_data_manager.get_dataset_dataloader("hdf5_copy")

total = 0
mask_num = 0

for entry in model_data_manager.data_information.loc[model_data_manager.data_information["dataset"] == 1].index:
    mask = loader.get_segmentation_mask(entry, "blood_vessel")
    mask_num += np.sum(mask)
    total += mask.shape[0] * mask.shape[1]

print(mask_num, " ", total, " ", mask_num / total)