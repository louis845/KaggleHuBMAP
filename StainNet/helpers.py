import importlib.resources

import torch
from .models import StainNet

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
