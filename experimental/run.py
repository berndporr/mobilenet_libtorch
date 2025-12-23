# Test locally via ExecuTorch runtime's pybind API (optional)
import torch
from executorch.runtime import Runtime
pte_filename = "mobilenet_features_quant.pte"
runtime = Runtime.get()
method = runtime.load_program(pte_filename).load_method("forward")
outputs = method.execute([torch.randn(1, 3, 224, 224)])
print(outputs)
