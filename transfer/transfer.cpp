#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "../mobilenet_v2.h"
#include <iostream>
#include <unistd.h>
#include <pwd.h>
#include <iostream>
#include <fstream>

namespace fs = std::filesystem;

// -------------------------
// Dataset implementation
// -------------------------
struct ImageFolderDataset : torch::data::Dataset<ImageFolderDataset>
{
    struct Sample
    {
        std::string image_path;
        int label;
    };

    std::vector<Sample> samples;

    ImageFolderDataset(const std::string &root, const std::vector<std::string> &classes)
    {
        for (size_t label = 0; label < classes.size(); label++)
        {
            fs::path class_path = fs::path(root) / classes[label];
            for (auto &p : fs::directory_iterator(class_path))
            {
                if (p.is_regular_file())
                {
                    samples.push_back({p.path().string(), (int)label});
                }
            }
        }
        std::cout << "Loaded " << samples.size() << " samples from " << root << "\n";
    }

    torch::data::Example<> get(size_t idx) override
    {
        const auto &sample = samples[idx];
        cv::Mat img = cv::imread(sample.image_path);
        if (img.empty())
        {
            throw std::runtime_error("Failed to load image: " + sample.image_path);
        }
        torch::Tensor data = MobileNetV2::preprocess(img);
        torch::Tensor label = torch::tensor(sample.label, torch::kLong);
        return {data, label};
    }

    torch::optional<size_t> size() const override
    {
        return samples.size();
    }
};

// location of the Kaggle dataset
const std::string root = "/.cache/kagglehub/datasets/abdalnassir/the-animalist-cat-vs-dog-classification/versions/1/Cat vs Dog/train/";

void progress(int epoch, int epochs, double loss)
{
    std::cout << "Epoch [" << epoch << "/" << epochs << "], Loss: "
              << loss << "\r" << std::flush;
}

// -------------------------
// Main training program
// -------------------------
int main()
{
    torch::manual_seed(42);
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        const torch::DeviceType device_type = torch::kCUDA;
        device = torch::Device(device_type);
    }

    std::vector<std::string> classes = {"Cat", "Dog"};
    std::string homedir(getpwuid(getuid())->pw_dir);
    ImageFolderDataset ds(homedir + root, classes);

    const size_t batch_size = 32;

    auto loader = torch::data::make_data_loader(
        ds.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size));

    // -------------------------
    // Model setup
    // -------------------------
    MobileNetV2 model;
    model.load_weights("../mobilenet_v2.pt");

    // replace the standard classifier by a custom one with
    // only two categories for cats and dogs
    model.classifier = torch::nn::Sequential(
        torch::nn::Dropout(0.2),
        torch::nn::Linear(model.getNinputChannels4Classifier(), classes.size()));
    model.register_module("custom", model.classifier);

    // Freeze the feature detectors
    for (auto &p : model.features->parameters())
        p.requires_grad_(false);

    // Optimizer only for classifier
    torch::optim::Adam optimizer(model.classifier->parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::CrossEntropyLoss criterion;

    model.to(device);

    // -------------------------
    // Training loop
    // -------------------------
    const size_t epochs = 5;

    std::fstream floss;
    floss.open("loss.dat",std::fstream::out);

    for (size_t epoch = 1; epoch <= epochs; epoch++)
    {
        float cumloss = 0;
        int n = 0;
        model.train();
        for (auto &batch : *loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            optimizer.zero_grad();
            auto output = model.forward(data);
            auto loss = criterion(output, target);
            loss.backward();
            optimizer.step();
            progress(epoch, epochs, loss.item<double>());
            cumloss += loss.item<double>();
            n++;
        }
	const double avgLoss = cumloss / (double)n;
        progress(epoch, epochs, avgLoss);
	floss << epoch << "\t" << avgLoss << std::endl;
        std::cout << std::endl;
    }
    std::cout << "Done.\n";
    return 0;
}
