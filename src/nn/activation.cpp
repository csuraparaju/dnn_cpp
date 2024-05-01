/**
 * @file activation.cpp
 * @author Krish Suraparaju (csurapar@andrew.cmu.edu)
 * @brief Provides the concrete implementation of the various activation
 *        functions defined in include/nn/activation.h header.
 * @version 0.1
 * @date 2024-05-01
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "../../include/nn/activation.h"


/**
 * @brief Computes the ReLU activation function on the input Z,
 *        storing the result in A. ReLU(Z) = max(0, Z).
 *
 * @param Z The input from the previous layer (before-activation)
 * @return Eigen::MatrixXd The activated output
 */
Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& Z) {
    this->A = A;
    A = Z.cwiseMax(0);
    return A;
}

/**
 * @brief Computes the derivative of the ReLU activation function,
 *       dL/dZ, where L is the loss and Z is the input.
 *
 * @param dLdA The derivative of the loss with respect to the activated output
 * @return Eigen::MatrixXd The derivative of the loss with respect to the input Z
 */
Eigen::MatrixXd ReLU::backward(const Eigen::MatrixXd& dLdA) {
    Eigen::MatrixXd curr_A = this->A;
    Eigen::MatrixXd dAdZ = (curr_A.array() > 0).cast<double>();
    return dLdA.cwiseProduct(dAdZ);
}

/**
 * @brief Computes the Sigmoid activation function on the input Z,
 *       storing the result in A. Sigmoid(Z) = 1 / (1 + exp(-Z)).
 *
 * @param Z The input from the previous layer (before-activation)
 * @return Eigen::MatrixXd The activated output (A)
 */
Eigen::MatrixXd Sigmoid::forward(const Eigen::MatrixXd& Z) {
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(Z.rows(), Z.cols());
    Eigen::MatrixXd A = 1 / (1 + (-Z.array()).exp());
    this->A = A;
    return A;
}

/**
 * @brief Computes the derivative of the Sigmoid activation function,
 *       dL/dZ, where L is the loss and Z is the input. Note that
 *       dA/dZ = A * (1 - A), since that is the derivative of the
 *       Sigmoid function.
 *
 * @param dLdA The derivative of the loss with respect to the activated output
 * @return Eigen::MatrixXd The derivative of the loss with respect to the input Z
 */
Eigen::MatrixXd Sigmoid::backward(const Eigen::MatrixXd& dLdA) {
    Eigen::MatrixXd A = this->A;
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(A.rows(), A.cols());
    Eigen::MatrixXd dAdZ = A.cwiseProduct(one - A);
    return dLdA.cwiseProduct(dAdZ);
}

/**
 * @brief Computes the Tanh activation function on the input Z,
 *       storing the result in A. Tanh(Z) = (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z)).
 *
 * @param Z The input from the previous layer (before-activation)
 * @return Eigen::MatrixXd The activated output (A)
 */
Eigen::MatrixXd Tanh::forward(const Eigen::MatrixXd& Z) {
    Eigen::MatrixXd A = (Z.array().exp() - (-Z.array()).exp()) / (Z.array().exp() + (-Z.array()).exp());
    this->A = A;
    return A;
}

/**
 * @brief Computes the derivative of the Tanh activation function,
 *       dL/dZ, where L is the loss and Z is the input. Note that
 *       dA/dZ = 1 - A^2, since that is the derivative of the
 *       Tanh function.
 *
 * @param dLdA The derivative of the loss with respect to the activated output
 * @return Eigen::MatrixXd The derivative of the loss with respect to the input Z
 */
Eigen::MatrixXd Tanh::backward(const Eigen::MatrixXd& dLdA) {
    Eigen::MatrixXd A = this->A;
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(A.rows(), A.cols());
    Eigen::MatrixXd dAdZ = one - A.cwiseProduct(A);
    return dLdA.cwiseProduct(dAdZ);
}

