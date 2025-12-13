#include "mobilenet_v2.h"
#include <opencv2/opencv.hpp>
#include <iostream>

const torch::Device device = torch::kCPU; // change to torch::kCUDA if available

const char pretrained_weights_file[] = "mobilenet_v2.pt";

const char labels_file[] = "labels.txt";

// loads the labels into a vector
const std::vector<std::string> loadLabels(const std::string filename)
{
    std::vector<std::string> lines;
    std::ifstream input(filename);
    for (std::string line; std::getline(input, line);)
    {
        lines.push_back(line);
    }
    return lines;
}

int inference(MobileNetV2& model, std::string path) {
    // load the image from disk
    cv::Mat img = cv::imread(path.c_str());
    if (img.empty()) throw std::invalid_argument("Cannot load image.");

    // scale and crop the image.
    torch::Tensor input = model.preprocess(img);

    // uploads the image to the device (CPU or GPU)
    input = input.to(device);

    // turn the image into a batch containing one image
    input = input.unsqueeze(0);

    // do the inference
    torch::Tensor output = model.forward(input);

    // getting rid of the batch and obtaining an array of label scores
    output = output.squeeze();

    // finding the label with the highest score and return its index
    return output.argmax(0).item<int>();
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <imagefile>\n", argv[0]);
        return -1;
    }

    // getting the text labels
    const auto labels = loadLabels(labels_file);

    // create an instance of MobilenetV2
    MobileNetV2 model;
    // load the pre-trained weights
    model.load_torchvision_weights(pretrained_weights_file);
    // upload it all to the device (CPU or GPU)
    model.to(device);
    // switch to pure inference so no training
    model.eval();

    // do the inference by loading an image and getting the index of the label
    const int idx = inference(model,argv[1]);
    std::cout << "Predicted class is " << idx << ": " << labels[idx] << std::endl;
    return 0;
}
