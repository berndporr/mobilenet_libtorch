# MobileNetV2 libtorch
C++ version of MobileNetV2 using libtorch which can import the pre-trained weights from torchvision.

Importing the weights instead of using the JIT has the advantage that one can do transfer learning also on an edge device, for example, to adapt to different situations locally.

This implementation follows very closely the [torchvision implementation of mobilenet v2](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py).

Mobilenet is described [here](https://arxiv.org/pdf/1801.04381).

Then simply include `mobilenet_v2.h` into your own project.

## Credit

(C) 2025 [Bernd Porr](https://www.berndporr.me.uk/), GPLv3
