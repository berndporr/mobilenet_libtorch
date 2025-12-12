#pragma once

#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <regex>
#include <filesystem>
#include <iostream>
#include <system_error>
#include <opencv2/opencv.hpp>

/***
 * MobileNetV2 C++ Implementation (LibTorch)
 * Which is able to load the pre-trained weights from torchvision
 * (c) 2025 Bernd Porr, GPLv3
 ***/

#ifdef NDEBUG
constexpr bool debugOutput = false;
#else
constexpr bool debugOutput = true;
#endif

/**
 * @brief Module which performs Convolution, Batch Norm and Relu6.
 * See https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L126
 */
struct Conv2dNormActivation : torch::nn::Module
{
    torch::nn::Sequential conv{nullptr};
    static constexpr char className[] = "Conv2dNormActivation";

    static inline torch::Tensor relu6(const torch::Tensor &x)
    {
        return torch::clamp(torch::relu(x), 0, 6);
    }

    Conv2dNormActivation(int in_channels,
                         int out_channels,
                         int kernel_size = 3,
                         int stride = 1,
                         int padding = -1,
                         int groups = 1)
    {
        const int dilation = 1;
        conv = torch::nn::Sequential();
        if (padding < 0)
        {
            padding = (kernel_size - 1) / 2 * dilation;
        }
        conv->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .groups(groups)
                .bias(false)));
        conv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)));
        conv->push_back(torch::nn::Functional(relu6));
        register_module(className, conv);
    }

    torch::Tensor forward(const torch::Tensor &x)
    {
        return conv->forward(x);
    }
};

/**
 * @brief Inverted residual
 * Converted from https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L19
 */
struct InvertedResidual : torch::nn::Module
{
    torch::nn::Sequential conv{nullptr};
    bool use_res_connect;
    static constexpr char className[] = "InvertedResidual";

    InvertedResidual(int inp, int oup, int stride, int expand_ratio)
    {
        if ((stride < 1) || (stride > 2))
        {
            throw std::invalid_argument("Stride needs to be 1 or 2.");
        }
        const int hidden_dim = (int)round(inp * expand_ratio);
        use_res_connect = (stride == 1) && (inp == oup);

        conv = torch::nn::Sequential();

        if (expand_ratio != 1)
        {
            conv->push_back(
                Conv2dNormActivation(inp,
                                     hidden_dim,
                                     /*kernel_size*/ 1));
        }

        conv->push_back(
            Conv2dNormActivation(hidden_dim,
                                 hidden_dim,
                                 /*kernel_size=*/3,
                                 /*stride=*/stride,
                                 /*padding=*/-1,
                                 /*groups=*/hidden_dim));

        conv->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(hidden_dim, oup,
                                     /*kernel_size=*/1)
                .stride(1)
                .padding(0)
                .bias(false)));
        conv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(oup)));

        register_module(className, conv);
    }

    torch::Tensor forward(const torch::Tensor &x)
    {
        if (use_res_connect)
        {
            return x + conv->forward(x);
        }
        else
        {
            return conv->forward(x);
        }
    }
};

/**
 * @brief Implementation of MobileNetV2 as done in py-torchvision
 * See: // https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L66
 */
struct MobileNetV2 : torch::nn::Module
{
    torch::nn::Sequential features{nullptr};
    torch::nn::Sequential classifier{nullptr}; // Dropout + Linear
    int features_output_channels = 1280;

    // MobileNetV2 inverted residual settings:
    // t, c, n, s  (expansion, output channels, repeats, stride)
    const std::vector<std::array<int, 4>> inverted_residual_setting = {
        {1, 16, 1, 1},
        {6, 24, 2, 2},
        {6, 32, 3, 2},
        {6, 64, 4, 2},
        {6, 96, 3, 1},
        {6, 160, 3, 2},
        {6, 320, 1, 1},
    };

    // Module name of the feature detector.
    static constexpr char featuresModuleName[] = "features";

    /**
     * @brief Name of the classifier module.
     * The name of the classifier module is needed whne replacing
     * the default classifier.
     */
    static constexpr char classifierModuleName[] = "classifier";

    /**
     * @brief Construct a new MobileNetV2 object
     * If you want to load the weights from torchvision into the classifier use the
     * default values for the parameters.
     *
     * @param num_classes Number of classes
     * @param width_mult Width multiplier - adjusts number of channels in each layer by this amount
     * @param round_nearest Round the number of channels in each layer to be a multiple of this number
     * @param dropout Dropout probability for the dropout layer in the classifier
     */
    MobileNetV2(int num_classes = 1000, float width_mult = 1.0f, int round_nearest = 8, float dropout = 0.2)
    {
        int input_channels = 32;
        input_channels = make_divisible(input_channels * width_mult, round_nearest);
        features_output_channels = make_divisible(features_output_channels * std::max(1.0f, width_mult), round_nearest);

        features = torch::nn::Sequential();

        features->push_back(
            Conv2dNormActivation(3,
                                 input_channels,
                                 /*kernel_size=*/3,
                                 /*stride =*/2));

        // inverted residual blocks
        for (const auto &cfg : inverted_residual_setting)
        {
            const int t = cfg[0];
            const int c = cfg[1];
            const int n = cfg[2];
            const int s = cfg[3];

            int output_channel = make_divisible(c * width_mult, round_nearest);
            for (int i = 0; i < n; ++i)
            {
                const int stride = (i == 0) ? s : 1;
                features->push_back(
                    InvertedResidual(input_channels, output_channel, stride, t));
                input_channels = output_channel;
            }
        }

        features->push_back(
            Conv2dNormActivation(input_channels,
                                 features_output_channels,
                                 /*kernel_size=*/1));

        register_module(featuresModuleName, features);

        // classifier: Dropout + Linear
        classifier = torch::nn::Sequential();
        classifier->push_back(torch::nn::Dropout(torch::nn::DropoutOptions(dropout)));
        classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(features_output_channels, num_classes)));
        register_module(classifierModuleName, classifier);
    }

    /**
     * @brief Gets the number of input channels for the classfier
     * This will make it easy to replace the classifier with anything the user wants
     * by creating their own torch::nn::Sequential() for the classifier.
     *
     * @return int The number of intput channels to the classifer class "classfier".
     */
    int getNinputChannels4Classifier() const
    {
        return features_output_channels;
    }

    /**
     * @brief Performs the forward pass
     *
     * @param x The batch of input images.
     * @return torch::Tensor The category scores for the different labels.
     */
    torch::Tensor forward(torch::Tensor x)
    {
        x = features->forward(x);
        const torch::nn::functional::AdaptiveAvgPool2dFuncOptions &ar = torch::nn::functional::AdaptiveAvgPool2dFuncOptions({1, 1});
        x = torch::nn::functional::adaptive_avg_pool2d(x, ar);
        x = torch::flatten(x, 1);
        x = classifier->forward(x);
        return x;
    }

    /**
     * @brief Initialize conv/bn/linear similar to torchvision defaults
     */
    void initialize_weights()
    {
        for (auto &module : modules(/*include_self=*/false))
        {
            if (auto M = dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
            {
                torch::nn::init::kaiming_normal_(M->weight, /*a=*/0, torch::kFanOut, torch::kReLU);
                if (M->options.bias())
                    torch::nn::init::zeros_(M->bias);
            }
            else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl *>(module.get()))
            {
                torch::nn::init::ones_(M->weight);
                torch::nn::init::zeros_(M->bias);
            }
            else if (auto M = dynamic_cast<torch::nn::LinearImpl *>(module.get()))
            {
                torch::nn::init::normal_(M->weight, 0.0, 0.01);
                torch::nn::init::zeros_(M->bias);
            }
        }
    }

    std::string ourkey2torchvision(std::string k) const
    {
        // called simply "conv" in the weights file
        k = std::regex_replace(k, std::regex(InvertedResidual::className), "conv");
        // not used at all in the weights file
        const std::string r = std::string(Conv2dNormActivation::className) + "\\.";
        k = std::regex_replace(k, std::regex(r), "");
        return k;
    }

    /**
     * @brief Loads a .pt weight file containing a dic with key/parameter pairs
     * See https://github.com/pytorch/pytorch/issues/36577
     *
     * @param pt filename of the .pt weight file
     */
    void load_weights(std::string pt)
    {
        std::ifstream input(pt, std::ios::binary);
        input.exceptions(input.failbit);
        std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));
        input.close();
        const c10::Dict<c10::IValue, c10::IValue> weights = torch::pickle_load(bytes).toGenericDict();
        if (debugOutput)
        {
            std::cerr << "Parameters we have in this model here: " << std::endl;
            for (auto const &m : named_parameters())
            {
                auto k = ourkey2torchvision(m.key());
                std::cerr << m.key() << "->" << k << ": " << m.value().sizes() << std::endl;
            }
            std::cerr << "Named buffers we have in this model here: " << std::endl;
            for (const auto &b : named_buffers())
            {
                auto k = ourkey2torchvision(b.key());
                std::cout << b.key() << "->" << k << ": " << b.value().sizes() << std::endl;
            }
            std::cerr << "Parameters we have in the weight file " << pt << ":" << std::endl;
            for (auto const &w : weights)
            {
                std::cerr << w.key() << ": " << w.value().toTensor().sizes() << std::endl;
            }
        }
        torch::NoGradGuard no_grad;
        if (debugOutput)
            std::cerr << "Loading weights" << std::endl;
        for (auto &m : named_parameters())
        {
            const std::string model_key = m.key();
            const std::string model_key4torchvision = ourkey2torchvision(model_key);
            if (debugOutput)
                std::cerr << "Searching for: " << model_key4torchvision << ": " << m.value().sizes() << std::endl;
            bool foundit = false;
            for (auto const &w : weights)
            {
                if (model_key4torchvision == w.key())
                {
                    if (debugOutput)
                        std::cerr << "Found it: " << w.key() << std::endl;
                    m.value().copy_(w.value().toTensor());
                    foundit = true;
                    break;
                }
            }
            if (!foundit)
                std::cerr << "Key: " << model_key4torchvision << " could not be found!" << std::endl;
        }
        if (debugOutput)
            std::cerr << "Loading named buffers" << std::endl;
        for (auto &b : named_buffers())
        {
            std::string model_key = b.key();
            std::string model_key4torchvision = ourkey2torchvision(model_key);
            if (debugOutput)
                std::cerr << "Searching for: " << model_key4torchvision << ": " << b.value().sizes() << std::endl;
            bool foundit = false;
            for (auto const &w : weights)
            {
                if (model_key4torchvision == w.key())
                {
                    if (debugOutput)
                        std::cerr << "Found it: " << w.key() << std::endl;
                    b.value().copy_(w.value().toTensor());
                    foundit = true;
                    break;
                }
            }
            if (!foundit)
                std::cerr << "Key: " << model_key4torchvision << " could not be found!" << std::endl;
        }
    }

    /**
     * @brief Preprocessing of an openCV image for inference or learning
     * The images are resized to 256x256, followed by a central crop of 224x224.
     * Finally the values are first rescaled to [0.0, 1.0]
     * and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
     *
     * @param img 8bit BGR openCV image with an aspect ratio of 1:1
     * @param resizeOnly If true the image is only resized to 224x224 but not cropped. Default: false.
     * @return torch::Tensor The image as a tensor ready to be used for inference and learning.
     */
    static torch::Tensor preprocess(cv::Mat img, bool resizeOnly = false)
    {
        constexpr int imageSizeBeforeCrop = 256;
        constexpr int finalImageSize = 224;
        constexpr int numChannels = 3; // colour

        if (img.depth() != CV_8U)
            throw std::invalid_argument("Image is not 8bit.");
        if (img.channels() != numChannels)
            throw std::invalid_argument("Image is not BGR / colour.");

        if (resizeOnly)
        {
            cv::resize(img, img, cv::Size(finalImageSize, finalImageSize));
        }
        else
        {
            cv::resize(img, img, cv::Size(imageSizeBeforeCrop, imageSizeBeforeCrop));
            constexpr int start = (imageSizeBeforeCrop - finalImageSize) / 2;
            const cv::Rect roi(start, start, finalImageSize, finalImageSize);
            img = img(roi).clone();
        }
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        torch::Tensor tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
        tensor = tensor.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);
        tensor = torch::data::transforms::Normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})(tensor);
        return tensor;
    }

    inline int make_divisible(int v, int divisor = 8, int min_value = -1) const
    {
        if (min_value < 0)
            min_value = divisor;
        int new_v = std::max(min_value, ((int)(((int)(v + divisor / 2)) / divisor)) * divisor);
        if (new_v < (0.9 * (float)v))
            new_v += divisor;
        return new_v;
    }
};
