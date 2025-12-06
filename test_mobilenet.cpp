#include "mobilenet_v2.h"
#include <opencv2/opencv.hpp>
#include <iostream>

torch::Tensor preprocess_opencv_bgr(const cv::Mat& img_bgr, int64_t target_w = 224, int64_t target_h = 224) {
    cv::Mat img;
    // Resize (preserve aspect? here simple resize)
    cv::resize(img_bgr, img, cv::Size(target_w, target_h));
    // Convert BGR -> RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Convert to float and to tensor
    torch::Tensor tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kUInt8);
    tensor = tensor.permute({2, 0, 1}).to(torch::kFloat32).div(255.0);

    // Normalize with ImageNet means/stds
    tensor[0] = (tensor[0] - 0.485f) / 0.229f; // R
    tensor[1] = (tensor[1] - 0.456f) / 0.224f; // G
    tensor[2] = (tensor[2] - 0.406f) / 0.225f; // B

    // Add batch
    tensor = tensor.unsqueeze(0);
    return tensor;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
	fprintf(stderr,"Usage: %s <imagefile>\n",argv[0]);
	return -1;
    }
    torch::manual_seed(1);
    torch::Device device = torch::kCPU; // change to torch::kCUDA if available

    // Construct MobileNetV2 with desired number of classes (default 1000)
    int64_t num_classes = 1000;
    MobileNetV2Impl model(num_classes, /*width_mult=*/1.0f);
    model.load_parameters("mobilenet_v2.pt");
    model.to(device);

    std::cout << "Model created.\n";

    // Example: replace classifier to new num_classes (e.g., 10)
    int64_t new_num_classes = 10;
    // safer: query last module in classifier
    auto& classifier_seq = model.classifier;
    // classifier: [Dropout, Linear], so classifier[1] is Linear
    auto linear_module = classifier_seq->ptr(1);
    auto linear = linear_module->as<torch::nn::Linear>();
    int64_t in_features = linear->options.in_features();

    // Create new linear and replace
    torch::nn::Linear new_fc(torch::nn::LinearOptions(in_features, new_num_classes));
    model.classifier = torch::nn::Sequential();
    model.classifier->push_back(torch::nn::Dropout(torch::nn::DropoutOptions(0.2)));
    model.classifier->push_back(new_fc);
    model.to(device);

    std::cout << "Replaced classifier with " << new_num_classes << " classes.\n";

    // Optionally freeze backbone
    for (auto& p : model.features->parameters()) p.requires_grad_(false);
    std::cout << "Feature extractor frozen.\n";

    // Example inference with OpenCV
    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) {
        std::cerr << "Failed to open test.jpg\n";
    } else {
        torch::Tensor input = preprocess_opencv_bgr(img).to(device);
        model.eval();
        torch::NoGradGuard no_grad;
        torch::Tensor logits = model.forward(input);
        auto pred = logits.argmax(1).item<int64_t>();
        std::cout << "Predicted class: " << pred << std::endl;
    }

    return 0;
}
