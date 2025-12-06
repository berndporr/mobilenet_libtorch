# mobilenet_libtorch
C++ version of mobilenet using libtorch (work in progress)

## Links

 - https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
 - https://docs.pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html
 - https://arxiv.org/pdf/1801.04381

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

Create the state dictionary which converts the weights from torchvision to a dictionary
which is readable by C++ (see https://github.com/pytorch/pytorch/issues/36577):

```
python create_dict_from_weights.py
```

The run the test which tries to classify a single image:

```
./test_mobilenet myimage.jpg
```
