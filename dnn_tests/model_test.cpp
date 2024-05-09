#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/nn/model.h"
#include "../include/nn/loss.h"
#include "../include/nn/activation.h"
#include <vector>
#include <utility>

using namespace Eigen;

TEST(LinearModel, Init) {
    size_t input_dim = 2;
    size_t output_dim = 3;
    size_t num_layers = 2; // Two layers, one input and one output

    std::unique_ptr<LossFunction> loss = std::make_unique<SoftmaxCrossEntropy>();
    std::vector<std::unique_ptr<ActivationFunction>> activations;
    activations.push_back(std::make_unique<ReLU>());
    activations.push_back(std::make_unique<Sigmoid>());

    // Make a new linear model
    LinearModel model(input_dim, output_dim, num_layers,
                    std::move(loss), std::move(activations));

}
