import importlib.resources
import os

import torch
import pandas as pd
import numpy as np
import h5py
import tqdm
import cv2
from .models import StainNet

import config

stainnet = None
def get_StainNet(device: torch.device):
    global stainnet
    if stainnet is None:
        stainnet_file = importlib.resources.path(__package__, "StainNet-Public_layer3_ch32.pth")
        stainnet = StainNet().to(device)
        stainnet.load_state_dict(torch.load(stainnet_file, map_location=torch.device("cpu"), weights_only=True))
        stainnet.eval()
    return stainnet

def stain_normalize(image: torch.Tensor, device: torch.device):
    stainnet = get_StainNet(device)
    with torch.no_grad():
        return stainnet(image)

def stain_normalize_image(image: np.ndarray, device: torch.device):
    image_tile = 2 * image.astype(np.float32) / 255 - 1
    normed_torch_tensor = stain_normalize(torch.from_numpy(image_tile).to(device).permute(2, 0, 1).unsqueeze(0), device)
    transformed_image_tile = ((normed_torch_tensor / 2 + 0.5) * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return transformed_image_tile


def construct_wsi(wsi_id):
    wsi_information = pd.read_csv(os.path.join(config.input_data_path, "tile_meta.csv"), index_col=0)
    wsi_information = wsi_information.loc[wsi_information["source_wsi"] == wsi_id]
    width = int(wsi_information["i"].max() + 512)
    height = int(wsi_information["j"].max() + 512)

    image_np = np.zeros((height, width, 3), dtype=np.uint8)

    for wsi_tile in wsi_information.index:
        x = wsi_information.loc[wsi_tile, "i"]
        y = wsi_information.loc[wsi_tile, "j"]
        image_tile = cv2.imread(os.path.join(config.input_data_path, "train", wsi_tile + ".tif"))
        image_tile = cv2.cvtColor(image_tile, cv2.COLOR_BGR2RGB)
        transformed_image_tile = stain_normalize_image(image_tile, config.device)
        image_np[y:y+512, x:x+512, :] = transformed_image_tile

    return image_np

folder = "stainnet_wsi"
if not os.path.isdir(folder):
    os.mkdir(folder)
class WSIImage:
    """Wrapper class for storing all StainNet WSI images."""
    def __init__(self):
        if not os.path.isfile(os.path.join(folder, "wsi_images.hdf5")):
            print()
            print("WSI images not found. Generating...")
            with h5py.File(os.path.join(folder, "wsi_images.hdf5"), "w") as f:
                for wsi_id in tqdm.tqdm(range(1, 15)):
                    if wsi_id != 5:
                        wsi_image_np = construct_wsi(wsi_id)
                        wsi_image_np_save = cv2.cvtColor(wsi_image_np, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(folder, "wsi_{}.png".format(wsi_id)), wsi_image_np_save)

                        f.create_dataset("wsi_{}".format(wsi_id), data=wsi_image_np, dtype=np.uint8, compression="gzip", compression_opts=4, chunks=(512, 512, 3))
                        del wsi_image_np

            print("Successfully generated WSI images.")

        self.wsi_images = h5py.File(os.path.join(folder, "wsi_images.hdf5"), "r")

    def get_whole_image(self, wsi_id):
        return np.array(self.wsi_images["wsi_{}".format(wsi_id)], dtype=np.uint8)

    def get_image(self, wsi_id, x1, x2, y1, y2):
        return np.array(self.wsi_images["wsi_{}".format(wsi_id)][y1:y2, x1:x2, :], dtype=np.uint8)

    def __del__(self):
        self.wsi_images.close()

images = WSIImage()