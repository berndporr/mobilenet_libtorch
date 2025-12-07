#pragma once

#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <regex>

/*
  MobileNetV2 C++ Implementation (LibTorch)
  - InvertedResidual block
  - MobileNetV2 module
*/

inline torch::Tensor relu6(const torch::Tensor &x)
{
    return torch::clamp(torch::relu(x), 0, 6);
}

// make_divisible mirrors torchvision.utils._make_divisible
inline int make_divisible(int v, int divisor = 8, int min_value = -1)
{
    if (min_value < 0)
        min_value = divisor;
    int new_v = std::max(min_value, (int)((int)(v + divisor / 2) / divisor * divisor));
    if (new_v < (0.9 * v))
        new_v += divisor;
    return new_v;
}

struct Conv2dNormActivation : torch::nn::Module
{
    torch::nn::Sequential conv{nullptr};

    Conv2dNormActivation(int in_channels,
			 int out_channels,
			 int kernel_size = 3,
			 int stride = 1,
			 int padding = -1,
			 int groups = 1,
			 bool linear = false)
    {
	const int dilation = 1;
	conv = torch::nn::Sequential();
	if (padding < 0) {
	    padding = (kernel_size - 1) / 2 * dilation;
	}
        conv->push_back(torch::nn::Conv2d(
			    torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
			    .stride(stride)
			    .padding(padding)
			    .groups(groups)
			    .bias(false)));
        conv->push_back(torch::nn::BatchNorm2d(out_channels));
        if (!linear) conv->push_back(torch::nn::Functional(relu6));
        register_module("Conv2dNormActivation", conv);
    }

    torch::Tensor forward(const torch::Tensor &x)
    {
	return conv->forward(x);
    }
};
    
struct InvertedResidual : torch::nn::Module
{
    torch::nn::Sequential conv{nullptr};
    bool use_res_connect;

    InvertedResidual(int inp, int oup, int stride, int expand_ratio)
    {
        use_res_connect = (stride == 1) && (inp == oup);
        int hidden_dim = inp * expand_ratio;

        conv = torch::nn::Sequential();

        if (expand_ratio != 1)
        {
	    conv->push_back(
		Conv2dNormActivation(inp,
				     hidden_dim,
				     /*kernel_size*/ 1,
				     /*stride*/ 1)
		);
        }
	
	// https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L126
	conv->push_back(
	    Conv2dNormActivation(hidden_dim,
				 hidden_dim,
				 /*kernel_size=*/ 3,
				 /*stride=*/ stride,
				 /*padding=*/ 1,
				 /*groups=*/ hidden_dim)
	    );
	
	conv->push_back(
	    Conv2dNormActivation(hidden_dim,
				 oup,
				 /*kernel_size=*/ 1,
				 /*stride=*/ 1,
				 /*padding=*/ -1,
				 /*groups=*/ 1,
				 true)
	    );
	
        register_module("InvertedResidual", conv);
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

	features->push_back("0",
			    Conv2dNormActivation(3,
						 input_channel,
						 /*kernel_size=*/ 3,
						 /*stride =*/ 2,
						 /*padding =*/ 1));
	
	int j = 1;
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
		std::cerr << "i=" << i << std::endl;
                features->push_back(std::to_string(j++),
				    InvertedResidual(input_channel, output_channel, stride, t));
                input_channel = output_channel;
            }
        }

	features->push_back(std::to_string(j++),
			    Conv2dNormActivation(input_channel,
						 last_channel,
						 /*kernel_size=*/ 1,
						 /*stride =*/ 1));

        register_module("features", features);

        // classifier: Dropout + Linear
        classifier = torch::nn::Sequential();
        classifier->push_back("0",torch::nn::Dropout(torch::nn::DropoutOptions(/*p=*/0.2)));
        classifier->push_back("1",torch::nn::Linear(torch::nn::LinearOptions(last_channel, num_classes)));
        register_module("classifier", classifier);

        // Initialize weights (same style as torchvision)
        _initialize_weights();
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = features->forward(x);
        const torch::nn::functional::AdaptiveAvgPool2dFuncOptions &ar = torch::nn::functional::AdaptiveAvgPool2dFuncOptions({1, 1});
        x = torch::nn::functional::adaptive_avg_pool2d(x, ar);
        x = x.view({x.size(0), -1});
        x = classifier->forward(x);
        return x;
    }

    void _initialize_weights()
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

    // https://github.com/pytorch/pytorch/issues/36577
    void load_parameters(std::string pt)
    {
        std::ifstream input(pt, std::ios::binary);
        std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));
        input.close();
        c10::Dict<c10::IValue, c10::IValue> weights = torch::pickle_load(bytes).toGenericDict();
        const torch::OrderedDict<std::string, at::Tensor> &model_params = named_parameters();
        std::vector<std::string> param_names;
        std::cerr << "Parameters we have in this model here: " << std::endl;
        for (auto const &w : model_params)
        {
	    std::string k = std::regex_replace( w.key(), std::regex("InvertedResidual"), "conv" );
	    k = std::regex_replace( k, std::regex("Conv2dNormActivation\\."), "");
            param_names.push_back(k);
            std::cerr << w.key() << "->" << k << ": " << w.value().sizes() << std::endl;
        }
        std::cerr << "Parameters we have in the weight file " << pt << ":" << std::endl;
        for (auto const &w : weights)
        {
            std::cerr << w.key() << ": " << w.value().toTensor().sizes() << std::endl;
        }
        torch::NoGradGuard no_grad;
	std::cerr << "Loading weights" << std::endl;
        for (auto const &w : weights)
        {
            std::string name = w.key().toStringRef();
            at::Tensor param = w.value().toTensor();
            if (std::find(param_names.begin(), param_names.end(), name) != param_names.end())
            {
                std::cerr << name << ": " << param.sizes() << std::endl;
                model_params.find(name)->copy_(param);
            }
            else
            {
                std::cout << name << " does not exist among model parameters." << std::endl;
            };
        }
    }
};
