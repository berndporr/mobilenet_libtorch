# MobileNetV2 libtorch
C++ version of MobileNetV2 using libtorch which can import the pre-trained weights from torchvision.

Importing the weights instead of using the JIT has the advantage that one can do transfer learning also on an edge device, for example, to adapt to different situations locally.

This implementation follows very closely the [torchvision implementation of mobilenet v2](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py).

Mobilenet is described [here](https://arxiv.org/pdf/1801.04381).

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

## Constructive classification example

Get the pretrained weights by running this python script:
```
python get_pretrained_weights.py
```
It downloads the weight file `mobilenet_v2-7ebf99e0.pth` from torchvision 
and converts its content into the state dict `mobilenet_v2.pt` 
which can then be loaded into the classifier via `load_weights`.

Run this demo which classifies a single image, for example this one:

```
./demo_mobilenet bird.jpg
```

![alt tag](bird.jpg)
[Phot credit: By Pierre-Selim - Flickr: Pica pica, CC BY-SA 2.0](https://commons.wikimedia.org/w/index.php?curid=19400996)

The output should look like this:

```
Predicted class 18: magpie
```

See also the subfolder `transfer` for transfer learning replacing
the standard classifier with a custom one and using the pre-trained weights
for the feature detector.

## Installation

All functionality is included in the single header file `mobilenet_v2.h`. To install it
just type:
```
sudo make install
```

Then simply include `mobilenet_v2.h` into your own project.

## Credit

(C) 2025 [Bernd Porr](https://www.berndporr.me.uk/), GPLv3
