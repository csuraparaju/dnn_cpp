#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/nn/model.h"
#include "../include/nn/loss.h"
#include "../include/nn/activation.h"
#include <vector>
#include <utility>
#include <iostream>

using namespace Eigen;


TEST(LinearModel, Forward) {

    class LinearModel : public Model {
        public:
            Linear linear1_ = Linear(2, 4); // Input layer
            Linear linear2_ = Linear(4, 2); // Output layer
            Sigmoid sig1_;
            Sigmoid sig2_;
            std::unique_ptr<LossFunction> loss_ = std::make_unique<SoftmaxCrossEntropy>();
            std::vector<std::unique_ptr<Layer>> layers_;
            std::vector<std::unique_ptr<ActivationFunction>> activations_;

            LinearModel()
                : Model(layers_, activations_, loss_) {
                layers_.emplace_back(std::make_unique<Linear>(linear1_));
                layers_.emplace_back(std::make_unique<Linear>(linear2_));
                activations_.emplace_back(std::make_unique<Sigmoid>(sig1_));
                activations_.emplace_back(std::make_unique<Sigmoid>(sig2_));
            }
    };

    LinearModel model;

    Eigen::MatrixXd X(4, 2);
    X << -4.0, -3.0,
         11.8, 3.2,
         -7.13, 1.56,
         0.132, 4.5896;

    Eigen::MatrixXd Y(4, 2);
    Y << 0.0, 1.0,
         0.0, 1.0,
         1.0, 0.0,
         0.0, 1.0;

    Eigen::MatrixXd A = model.forward(X);

    std::cout << A << std::endl;
}
