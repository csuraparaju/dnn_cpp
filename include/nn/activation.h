/**
 * @file activation.h
 * @author Krish Suraparaju (csurapar@andrew.cmu.edu)
 * @brief Provides the interface for the various activation functions
 * used in the neural network.
 *
 * Activation functions are used to introduce non-linearity in the neural network.
 * The primary purpose of having nonlinear components in the neural network (fNN)
 * is to allow it to approximate nonlinear functions. Without activation functions,
 * fNN will always be linear, no matter how deep it is.
 *
 * The forward activation takes in Z -> the result of transforming an input
 * through some layer. It returns A, the activated version of this.
 *
 * The backwards function takes in dLdA, the derivative of loss with respect to
 * the output of the layer. This signifies however much our loss changes based on
 * change in the output.
 *
 * By multiplying dLdA with dAdZ, we get dLdZ, the change in loss with respect to
 * the input. This is then passed to the layer.
 *
 * @version 0.1
 * @date 2024-05-01
 *
 * @copyright Copyright (c) 2024
 *
 */


#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <Eigen/Dense>

// Base ActivationFunction class
class ActivationFunction {
public:
    /**
     * @brief Stores the activated output after a forward pass. Used later
     *        in the backward pass to compute the gradients.
     *
     */
    Eigen::MatrixXd A;
    /**
     * @brief Construct a new Activation Function object
     *
     */
    ActivationFunction() {}

    /**
     * @brief Destroy the Activation Function object
     *
     */
    virtual ~ActivationFunction() {}

    /**
     * @brief Applies the activation function to the input Z
     *
     * @param Z The input from the previous layer (before-activation)
     * @return A Eigen::MatrixXd The activated output
     */
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& Z) = 0;

    /**
     * @brief Computes the derivative of the activation function,
     *        dL/dZ, where L is the loss and Z is the input. Note that
     *        dL/dZ = dL/dA * dA/dZ, which is why we need to store
     *        the A value from the forward pass.
     *
     * @param dLdA The derivative of the loss with respect to the activated output
     * @return Eigen::MatrixXd The derivative of the loss with respect to the input Z
     */
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& dLdA) = 0;
};

// Concrete class for the ReLU activation function
class ReLU : public ActivationFunction {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& Z) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& dLdA) override;
};

// Concrete class for the Sigmoid activation function
class Sigmoid : public ActivationFunction {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& Z) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& dLdA) override;
};

// Concrete class for the Tanh activation function
class Tanh : public ActivationFunction {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& Z) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& dLdA) override;
};

#endif // ACTIVATION_H
