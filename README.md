# mobilenet libtorch
C++ version of mobilenet using libtorch which can use the pre-trained weights from torchvision.

This C++ implementation follows very closely the torchvision one: 
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

Mobilenet is described here: https://arxiv.org/pdf/1801.04381

## Prerequisites Libraries and packages

1) Make sure you have `cmake` and a c++ compiler installed.

2) Libtorch

 - Intel architectures: Get libtorch from the [PyTorch homepage](https://pytorch.org/get-started/locally/). Create an environment variable `CMAKE_PREFIX_PATH=/path/to/libtorch` pointing to the libtorch directory.
 - ARM Debian (Raspberry PI): just do `apt install libtorch-dev` and you are all set!

## How to compile

Type:

```
cmake .
```
to create the makefile and then

```
make
```
to compile the library and the demos.

## How to run

Get the pretrained weights:
```
python get_pretrained_weights.py
```
This script downloads the weight file `mobilenet_v2-7ebf99e0.pth` from torchvision and converts its content 
into a state dict `mobilenet_v2.pt` which can then be loaded into libtorch.

Run this demo which classifies a single image, for example this one:

```
./demo_mobilenet bird.jpg
```

![alt tag](bird.jpg)
Phot credit: By Pierre-Selim - Flickr: Pica pica, CC BY-SA 2.0, https://commons.wikimedia.org/w/index.php?curid=19400996

The output should look like this:

```
Predicted class 18: magpie
```

# Credit

(C) 2025 Bernd Porr, GPLv3
