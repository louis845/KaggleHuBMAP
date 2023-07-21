import importlib.resources
import os

import torch
import pandas as pd
import numpy as np
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

def construct_wsi(wsi_id):
    wsi_information = pd.read_csv(os.path.join(config.input_data_path, "tile_meta.csv"), index_col=0)
    wsi_information = wsi_information.loc[wsi_information["source_wsi"] == wsi_id]
    width = int(wsi_information["i"].max() + 512)
    height = int(wsi_information["j"].max() + 512)

    image_np = np.zeros((height, width, 3), dtype=np.uint8)

    for wsi_tile in wsi_information.index:
        x = wsi_information.loc[wsi_tile, "i"]
        y = wsi_information.loc[wsi_tile, "j"]
        image_tile = cv2.imread(os.path.join(config.input_data_path, "train", wsi_tile + ".png"))
        image_tile = cv2.cvtColor(image_tile, cv2.COLOR_BGR2RGB)
        image_tile = 2 * image_tile.astype(np.float32) / 255 - 1
        normed_torch_tensor = stain_normalize(torch.from_numpy(image_tile).to(config.device).permute(2, 0, 1).unsqueeze(0), config.device)
        transformed_image_tile = ((normed_torch_tensor / 2 + 0.5) * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        image_np[y:y+512, x:x+512, :] = transformed_image_tile

    return image_np