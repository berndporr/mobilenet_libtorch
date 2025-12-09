#pragma once

#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <regex>

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

inline torch::Tensor relu6(const torch::Tensor &x)
{
    return torch::clamp(torch::relu(x), 0, 6);
}

// make_divisible mirrors torchvision.utils._make_divisible
inline int make_divisible(int v, int divisor = 8, int min_value = -1)
{
    if (min_value < 0)
        min_value = divisor;
    int new_v = std::max(min_value, ((int)(((int)(v + divisor / 2)) / divisor)) * divisor);
    if (new_v < (0.9 * (float)v))
        new_v += divisor;
    return new_v;
}

// https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L126
struct Conv2dNormActivation : torch::nn::Module
{
    torch::nn::Sequential conv{nullptr};
    static constexpr char className[] = "Conv2dNormActivation";

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
        conv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels).track_running_stats(true)));
        conv->push_back(torch::nn::Functional(relu6));
        register_module(className, conv);
    }

    torch::Tensor forward(const torch::Tensor &x)
    {
        return conv->forward(x);
    }
};

// https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L19
struct InvertedResidual : torch::nn::Module
{
    torch::nn::Sequential conv{nullptr};
    bool use_res_connect;
    static constexpr char className[] = "InvertedResidual";

    InvertedResidual(int inp, int oup, int stride, int expand_ratio)
    {
        if ((stride < 1) || (stride > 2))
        {
            throw "Stride needs to be 1 or 2.";
        }
        int hidden_dim = (int)round(inp * expand_ratio);
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
        conv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(oup).track_running_stats(true)));

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

// https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L66
struct MobileNetV2 : torch::nn::Module
{
    torch::nn::Sequential features{nullptr};
    torch::nn::Sequential classifier{nullptr}; // Dropout + Linear
    int last_channel;

    MobileNetV2(int num_classes = 1000, float width_mult = 1.0f, int round_nearest = 8)
    {
        // MobileNetV2 inverted residual settings:
        // t, c, n, s  (expansion, output channels, repeats, stride)
        std::vector<std::array<int, 4>> inverted_residual_setting = {
            {1, 16, 1, 1},
            {6, 24, 2, 2},
            {6, 32, 3, 2},
            {6, 64, 4, 2},
            {6, 96, 3, 1},
            {6, 160, 3, 2},
            {6, 320, 1, 1},
        };

        int input_channel = 32;
        last_channel = 1280;

        input_channel = make_divisible(input_channel * width_mult, round_nearest);
        last_channel = make_divisible(last_channel * std::max(1.0f, width_mult), round_nearest);

        features = torch::nn::Sequential();

        features->push_back(
            Conv2dNormActivation(3,
                                 input_channel,
                                 /*kernel_size=*/3,
                                 /*stride =*/2));

        // inverted residual blocks
        for (const auto &cfg : inverted_residual_setting)
        {
            int t = cfg[0];
            int c = cfg[1];
            int n = cfg[2];
            int s = cfg[3];

            int output_channel = make_divisible(c * width_mult, round_nearest);
            for (int i = 0; i < n; ++i)
            {
                int stride = (i == 0) ? s : 1;
                features->push_back(
                    InvertedResidual(input_channel, output_channel, stride, t));
                input_channel = output_channel;
            }
        }

        features->push_back(
            Conv2dNormActivation(input_channel,
                                 last_channel,
                                 /*kernel_size=*/1));

        register_module("features", features);

        // classifier: Dropout + Linear
        classifier = torch::nn::Sequential();
        classifier->push_back(torch::nn::Dropout(torch::nn::DropoutOptions(/*p=*/0.2)));
        classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(last_channel, num_classes)));
        register_module("classifier", classifier);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = features->forward(x);
        const torch::nn::functional::AdaptiveAvgPool2dFuncOptions &ar = torch::nn::functional::AdaptiveAvgPool2dFuncOptions({1, 1});
        x = torch::nn::functional::adaptive_avg_pool2d(x, ar);
        x = torch::flatten(x, 1);
        x = classifier->forward(x);
        return x;
    }

    void initialize_weights()
    {
        // Initialize conv/bn/linear similar to torchvision defaults
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

    // renames the keys here so that they match the python torchvision naming convention
    std::string ourkey2torchvision(std::string k)
    {
        // called simply "conv" in the weights file
        k = std::regex_replace(k, std::regex(InvertedResidual::className), "conv");
        // not used at all in the weights file
        const std::string r = std::string(Conv2dNormActivation::className) + "\\.";
        k = std::regex_replace(k, std::regex(r), "");
        return k;
    }

    // https://github.com/pytorch/pytorch/issues/36577
    void load_weights(std::string pt)
    {
        std::ifstream input(pt, std::ios::binary);
        std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));
        input.close();
        c10::Dict<c10::IValue, c10::IValue> weights = torch::pickle_load(bytes).toGenericDict();
        const torch::OrderedDict<std::string, at::Tensor> &model_params = named_parameters();
        if (debugOutput)
        {
            std::cerr << "Parameters we have in this model here: " << std::endl;
            for (auto const &m : model_params)
            {
                auto k = ourkey2torchvision(m.key());
                std::cerr << m.key() << "->" << k << ": " << m.value().sizes() << std::endl;
            }
            std::cerr << "Named buffers we have in this model here: " << std::endl;
            for (auto &b : named_buffers())
            {
                std::cout << b.key() << "\n";
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
        for (auto &m : model_params)
        {
            std::string model_key = m.key();
            std::string model_key4torchvision = ourkey2torchvision(model_key);
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
};
