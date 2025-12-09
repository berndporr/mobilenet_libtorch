# Loads the weight file from torchvision and converts it to a dict
import torch
import requests
from io import BytesIO
url = "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth"
with requests.get(url, stream=True) as response:
    response.raise_for_status()
    buffer = BytesIO(response.content)
    dict = torch.load(buffer)
    for param_tensor in dict:
        print(param_tensor, "\t", dict[param_tensor].size())
    w = {k: v for k, v in dict.items()}
    torch.save(w, "mobilenet_v2.pt")
