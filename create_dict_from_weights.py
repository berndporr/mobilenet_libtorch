# Loads the weight file from torchvision and converts it to a dict
import torch
dict = torch.load("mobilenet_v2-b0353104.pth")
for param_tensor in dict:
    print(param_tensor, "\t", dict[param_tensor].size())
w = {k: v for k, v in dict.items()}
torch.save(w, "mobilenet_v2.pt")
