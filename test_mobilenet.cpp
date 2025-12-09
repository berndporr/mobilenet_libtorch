#include "mobilenet_v2.h"
#include <opencv2/opencv.hpp>
#include <iostream>

torch::Tensor preprocess_opencv_bgr(cv::Mat img, int target_w = 224, int target_h = 224)
{
    cv::resize(img, img, cv::Size(target_w, target_h));
    // Convert BGR -> RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Convert to float and to tensor
    torch::Tensor tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    tensor = tensor.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);
    tensor = torch::data::transforms::Normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})(tensor);
    return tensor;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <imagefile>\n", argv[0]);
        return -1;
    }
    
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
    torch::Tensor input = preprocess_opencv_bgr(img).to(device);

    input = input.unsqueeze(0);
    torch::Tensor output = model.forward(input);
    output = output.squeeze();
    
    auto idx = output.argmax(0).item<int>();
    std::cout << "Predicted class: " << idx << " " << std::endl;

    return 0;
}
