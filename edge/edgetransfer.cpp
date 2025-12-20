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

// Path of the Kaggle dataset
const fs::path datasetpath = ".cache/kagglehub/datasets/abdalnassir/the-animalist-cat-vs-dog-classification/versions/1/Cat vs Dog/train/";

// Path to the pretrained weights file
const char pretrained_weights_file[] = "../mobilenet_v2.pt";

// Path to the loss log file
const char loss_file[] = "loss.dat";

// Subdirs of the two classes
const std::vector<fs::path> classes = {"Cat", "Dog"};

// The batch size for training
const int batch_size = 32;

// The number of epochs
const int epochs = 50;

// -------------------------
// Dataset implementation
// -------------------------
struct ImageFolderDataset : torch::data::Dataset<ImageFolderDataset>
{
    struct Sample
    {
        fs::path image_path;
        int label;
    };

    std::vector<Sample> samples;

    ImageFolderDataset(const fs::path &root, const std::vector<fs::path> &classes)
    {
        for (size_t label = 0; label < classes.size(); label++)
        {
            const fs::path class_path = root / classes[label];
            for (const auto &p : fs::directory_iterator(class_path))
            {
                if (p.is_regular_file())
                {
                    samples.push_back({p.path(), (int)label});
                }
            }
        }
        std::cout << "Loaded " << samples.size() << " samples from " << datasetpath.string() << "\n";
    }

    torch::data::Example<> get(size_t idx) override
    {
        const auto &sample = samples[idx];
        const cv::Mat img = cv::imread(sample.image_path.string());
        if (img.empty())
        {
            throw std::runtime_error("Failed to load image: " + sample.image_path.string());
        }
        const torch::Tensor data = MobileNetV2::preprocess(img);
        const torch::Tensor label = torch::tensor(sample.label, torch::kLong);
        return {data, label};
    }

    torch::optional<size_t> size() const override
    {
        return samples.size();
    }
};

// Simple progress output on the same line. No new line.
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

    const fs::path homedir(getpwuid(getuid())->pw_dir);
    ImageFolderDataset ds(homedir / datasetpath, classes);

    // Creates a DataLoader instance for a stateless dataset.
    // The sampler is RandomSampler so shuffling is enabled.
    auto loader = torch::data::make_data_loader(
        ds.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size));

    // Model setup
    MobileNetV2 model;

    // Load the pre-trained weights.
    model.load_torchvision_weights(pretrained_weights_file);

    // Replace the standard classifier by this custom one with
    // only two categories for cats and dogs.
    auto newClassifier = torch::nn::Sequential(
        torch::nn::Dropout(0.2),
        torch::nn::Linear(model.getNinputChannelsOfClassifier(), classes.size()));
    model.replaceClassifier(newClassifier);

    // Freeze the feature detectors.
    model.setFeaturesLearning(false);

    // Optimizer only for classifier.
    torch::optim::Adam optimizer(model.getClassifier()->parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::CrossEntropyLoss criterion;

    // Send the model to the CPU or GPU
    model.to(device);

    // Logging of the loss
    std::fstream floss;
    floss.open(loss_file, std::fstream::out);

    // Training loop
    for (int epoch = 1; epoch <= epochs; epoch++)
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
        std::cout << std::endl
                  << std::flush;
    }
    std::cout << "Done.\n";
    return 0;
}
