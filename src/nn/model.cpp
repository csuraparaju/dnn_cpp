/**
 * @file model.cpp
 * @author your name (you@domain.com)
 * @brief Provides the concrete implementation of the various neural network models
 * defined in include/nn/model.h header.
 * @version 0.1
 * @date 2024-05-08
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "../../include/nn/model.h"
#include <Eigen/Dense>

/**
 * @brief Implements the generic forward pass of the model.
 * Calls the forward function of each layer and activation function
 * in the model.
 *
 * @param X
 * @return Eigen::MatrixXd
 */
Eigen::MatrixXd Model::forward(const Eigen::MatrixXd& X) {
    Eigen::MatrixXd A = X;
    for (int i = 0; i < this->layers.size(); i++) {
        A = this->layers[i]->forward(A);

        // Guard check if activations are not provided
        if(i > this->activations.size() - 1) {
            return A;
        }

        A = this->activations[i]->forward(A);
    }
    return A;
}

/**
 * @brief Implements the generic backward pass of the model.
 * Calls the backward function of each layer and activation function
 * in the model.
 *
 * @param dLdZ
 */
void Model::backward() {
    Eigen::MatrixXd dLdA = this->loss->backward();
    Eigen::MatrixXd dLdZ;
    for (int i = this->layers.size() - 1; i >= 0; i--) {
        if(i <= this->activations.size() - 1) {
            dLdZ = this->activations[i]->backward(dLdA);
        }
        dLdA = layers[i]->backward(dLdA);
    }
}