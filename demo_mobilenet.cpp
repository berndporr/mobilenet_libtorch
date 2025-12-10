#include "mobilenet_v2.h"
#include <opencv2/opencv.hpp>
#include <iostream>

std::vector<std::string> loadLabels(std::string filename)
{
    std::vector<std::string> lines;
    std::ifstream input(filename);
    for (std::string line; std::getline(input, line);)
    {
        lines.push_back(line);
    }
    return lines;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <imagefile>\n", argv[0]);
        return -1;
    }

    auto labels = loadLabels("labels.txt");

    torch::manual_seed(1);
    torch::Device device = torch::kCPU; // change to torch::kCUDA if available

    MobileNetV2 model;
    model.load_weights("mobilenet_v2.pt");
    model.to(device);
    model.eval();

    cv::Mat img = cv::imread(argv[1]);
    if (img.empty())
    {
        std::cerr << "Failed to open the image.\n";
        return -1;
    }
    torch::Tensor input = model.preprocess(img).to(device);

    input = input.unsqueeze(0);
    torch::Tensor output = model.forward(input);
    output = output.squeeze();

    auto idx = output.argmax(0).item<int>();
    std::cout << "Predicted class " << idx << ": " << labels[idx] << std::endl;

    return 0;
}
