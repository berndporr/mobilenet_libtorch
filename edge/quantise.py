import kagglehub
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
train_batch_size = 30
eval_batch_size = 50

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torchao.quantization.pt2e'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)

def load_model():
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
    model.to("cpu")
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

def prepare_data_loaders(data_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),     # Standard for CNNs
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],     # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(
        root=data_path+"/train/",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=data_path+"/train/",
        transform=transform
    )

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_batch_size,
        sampler=test_sampler)

    return train_data_loader, test_data_loader

def calibrate(model, data_loader):
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

data_path = kagglehub.dataset_download("abdalnassir/the-animalist-cat-vs-dog-classification")+"/Cat vs Dog"
print("File are here: ",data_path)
data_loader, data_loader_test = prepare_data_loaders(data_path)
example_inputs = (next(iter(data_loader))[0])
criterion = nn.CrossEntropyLoss()
float_model = load_model().to("cpu")
float_model.eval()

# create another instance of the model since
# we need to keep the original model around
model_to_quantize = load_model().to("cpu")

model_to_quantize.eval()

example_inputs = (torch.rand(32, 3, 224, 224),)
exported_model = torch.export.export(model_to_quantize, example_inputs).module()

dynamic_shapes = tuple(
  {0: torch.export.Dim("dim", min=1, max=1024)} if i == 0 else None
  for i in range(len(example_inputs))
)

print("Dynshapes:",dynamic_shapes)

exported_model = torch.export.export(model_to_quantize, example_inputs, dynamic_shapes=dynamic_shapes).module()

quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())

prepared_model = prepare_pt2e(exported_model, quantizer)

calibrate(prepared_model, data_loader_test)  # run calibration on sample data

quantized_model = convert_pt2e(prepared_model)
print(quantized_model)

# Baseline model size
print("Size of baseline model")
print_size_of_model(float_model)

# Quantized model size
print("Size of model after quantization")
# export again to remove unused weights
quantized_model = torch.export.export(quantized_model, example_inputs).module()
print_size_of_model(quantized_model)

# capture the model to get an ExportedProgram
quantized_ep = torch.export.export(quantized_model, example_inputs)

output_path = torch._inductor.aoti_compile_and_package(
        quantized_ep,
        package_path=os.path.join(os.getcwd(), "model.pt2"),
    )
