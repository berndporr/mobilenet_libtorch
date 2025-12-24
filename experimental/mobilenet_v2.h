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
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

/***
 * MobileNetV2 C++ Implementation (LibTorch).
 * It's able to load pre-trained weights from torchvision
 * and has the neccessary methods to enable transfer learning.
 * (c) 2025 Bernd Porr, GPLv3.
 ***/

#ifdef NDEBUG
constexpr bool debugOutput = false;
#else
constexpr bool debugOutput = true;
#endif

/**
 * @brief Implementation of MobileNetV2 as done in py-torchvision
 * See: // https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L66
 */
class MobileNetV2 : public torch::nn::Module
{
public:
    /**
     * @brief Construct a new MobileNetV2 object.
     * If you want to load the weight files from torchvision into this class use the
     * default values for the parameters.
     *
     * @param num_classes Number of classes.
     * @param width_mult Width multiplier - adjusts number of channels in each layer by this amount.
     * @param round_nearest Round the number of channels in each layer to be a multiple of this number.
     * @param dropout Dropout probability for the dropout layer in the classifier.
     */
    MobileNetV2(int num_classes = 1000, float width_mult = 1.0f, int round_nearest = 8, float dropout = 0.2)
    {
        quantFeatures = std::make_shared<executorch::extension::Module>("model.pte");

        executorch::runtime::Error error = quantFeatures->load();
        if (!(quantFeatures->is_loaded()))
        {
            std::cerr << "Err:" << (int)error << std::endl;
            exit(-1);
        }

        int input_channels = 32;
        input_channels = make_divisible(input_channels * width_mult, round_nearest);
        features_output_channels = make_divisible(features_output_channels * std::max(1.0f, width_mult), round_nearest);

        //        register_module(featuresModuleName, quantFeatures);

        // classifier: Dropout + Linear
        classifier = torch::nn::Sequential();
        classifier->push_back(torch::nn::Dropout(torch::nn::DropoutOptions(dropout)));
        classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(features_output_channels, num_classes)));
        register_module(classifierModuleName, classifier);
    }

    /**
     * @brief Name of the features submodule.
     * This appears as part of the key in named_parametes and named_buffers.
     */
    static constexpr char featuresModuleName[] = "features";

    /**
     * @brief Name of the classifier submodule.
     * This appears as part of the key in named_parametes and named_buffers.
     */
    static constexpr char classifierModuleName[] = "classifier";

    /**
     * @brief Performs the forward pass.
     *
     * @param x The batch of input images.
     * @return torch::Tensor The category scores for the different labels.
     */
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.contiguous().cpu();
        std::vector<int> insizes(
            x.sizes().begin(),
            x.sizes().end());
        auto et_tensor = executorch::extension::from_blob(
            x.data_ptr<float>(),
            insizes);
        executorch::runtime::EValue input = executorch::runtime::EValue(et_tensor);
        auto result = quantFeatures->forward({input});
        if (!result.ok())
        {
            std::cerr << "Fatal. No result from model." << std::endl;
            exit(1);
        }
        const auto et = result->at(0).toTensor();
        std::vector<long int> outsizes(
            et.sizes().begin(),
            et.sizes().end());
        x = torch::from_blob(
            et.data_ptr<float>(), // raw pointer
            outsizes,
            torch::TensorOptions()
                .dtype(torch::kFloat)
                .device(torch::kCPU));
        const torch::nn::functional::AdaptiveAvgPool2dFuncOptions &ar = torch::nn::functional::AdaptiveAvgPool2dFuncOptions({1, 1});
        x = torch::nn::functional::adaptive_avg_pool2d(x, ar);
        x = torch::flatten(x, 1);
        x = classifier->forward(x);
        return x;
    }

    /**
     * @brief Initialize conv/bn/linear similar to torchvision defaults.
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

    /**
     * @brief Loads a .pt weight file containing a dict with key/parameter pairs.
     * See https://github.com/pytorch/pytorch/issues/36577
     * The difference between pytorch and libtorch is that pytorch just has
     * named parameters but libtorch has both named parameters and named buffers.
     * This method makes sure that the key/parameters pairs are loaded into
     * both named buffers and named parameters.
     *
     * @param pt filename of the .pt weight file.
     */
    void load_torchvision_weights(std::string pt)
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
     * @brief Preprocessing of an openCV image for inference or learning.
     * The images are resized to 256x256, followed by a central crop of 224x224.
     * Finally the values are first rescaled to [0.0, 1.0]
     * and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
     *
     * @param img 8bit BGR openCV image with an aspect ratio of 1:1.
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

    /**
     * @brief Gets the number of input channels of the classifier.
     * This will make it easy to replace the classifier with anything the user wants
     * by creating their own torch::nn::Sequential() for the classifier.
     *
     * @return int The number of intput channels of classifier class "classfier".
     */
    int getNinputChannelsOfClassifier() const
    {
        return features_output_channels;
    }

    /**
     * @brief Replaces classifier with a new one.
     *
     * For transfer learning the default classifier is replaced with a new one.
     * @param newClassifier The new classifier.
     */
    void replaceClassifier(torch::nn::Sequential &newClassifier)
    {
        classifier = newClassifier;
        replace_module(MobileNetV2::classifierModuleName, newClassifier);
    }

    /**
     * @brief Gets the Classifier object.
     *
     * Gets a shared pointer to the classifier, for example, to attach an optimiser
     * for transfer learning.
     *
     * @return torch::nn::Sequential
     */
    torch::nn::Sequential getClassifier() const
    {
        return classifier;
    }

private:
    /**
     * @brief Classifier submodule.
     */
    torch::nn::Sequential classifier{nullptr};

    // Features output channels but can be scaled.
    int features_output_channels = 1280;

    // Helper which maps the libtorch keys to pytorch keys.
    // libtorch requires names for the submodules, for example:
    // features.14.InvertedResidual.1.Conv2dNormActivation.1.weight.
    // However, pytorch has no names for the submodules and needs to be removed:
    // features.14.conv.1.1.weight.
    // Also it renames "InvertedResidual" to "conv" which is just due to my
    // choice to call it what it is and not just "conv".
    std::string ourkey2torchvision(std::string k) const
    {
        return k;
    }

    // Makes a value divisible.
    inline int make_divisible(int v, int divisor = 8, int min_value = -1) const
    {
        if (min_value < 0)
            min_value = divisor;
        int new_v = std::max(min_value, ((int)(((int)(v + divisor / 2)) / divisor)) * divisor);
        if (new_v < (0.9 * (float)v))
            new_v += divisor;
        return new_v;
    }

    std::shared_ptr<executorch::extension::Module> quantFeatures;
};
