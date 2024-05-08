/**
 * @file layer.h
 * @author Krish Suraparaju (csurapar@andrew.cmu.edu)
 * @brief
 *
 * Layers are the basic building blocks of a neural network. Each layer
 * consists of a set of neurons that process input data and pass the output
 * to the next layer. The output of a layer is computed using a set of weights
 * and biases that are learned during the training process. The weights and
 * biases are updated using an optimization algorithm such as gradient descent.
 *
 * Currently, the following layers are implemented:
 * 1. Linear Layer - Applies a linear transformation to the incoming data.
 *                   The output is computed as Z = A * W^T + ι_N * b.
 *
 * @version 0.1
 * @date 2024-05-01
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <Eigen/Dense>

#ifndef LAYER_H
#define LAYER_H
class Layer {
public:
    Eigen::MatrixXd W;  // Weights
    Eigen::MatrixXd b;  // Biases
    Eigen::MatrixXd A;  // Activated output from previous layer
    Eigen::MatrixXd dLdW; // Gradient of loss w.r.t. weights
    Eigen::MatrixXd dLdb; // Gradient of loss w.r.t. biases
    size_t N;  // Number of samples
    size_t in_size;  // Size of input
    size_t out_size; // Size of output

    /**
     * @brief Construct a new Layer object
     *
     * @param input_size The size of the input to the layer
     * @param output_size The size of the output from the layer
     */
    Layer(size_t input_size, size_t output_size) {
        this->W = Eigen::MatrixXd::Random(output_size, input_size);
        this->b = Eigen::MatrixXd::Random(output_size, 1);
        this->N = 0;
        this->in_size = input_size;
        this->out_size = output_size;
    }

    /**
     * @brief Destroy the Layer object
     *
     */
    virtual ~Layer() {}

    /**
     * @brief Fowrard pass through the layer, specific to each layer type
     *
     * @param A
     * @return Eigen::MatrixXd
     */
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& A) = 0;

    /**
     * @brief Backward pass through the layer, specific to each layer type
     *
     * @param dLdZ
     * @return Eigen::MatrixXd
     */
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& dLdZ) = 0;
};

class Linear : public Layer {
public:
    Linear(size_t input_size, size_t output_size) : Layer(input_size, output_size) {}
    /**
     * @brief During forward propagation, we apply a linear transformation
     * to the incoming data A to obtain output data Z using a weight matrix
     * W and a bias vector b. That is, Z = A * W^T + ι_N * b. The variable
     * ι_N is a column vector of ones of size N (the batch size), and is used
     * to broadcast the bias vector b across all samples in the batch.
     *
     * @param A The input to the layer
     * @return Eigen::MatrixXd The output of the layer
    */
    Eigen::MatrixXd forward(const Eigen::MatrixXd& A) override;

    /**
     * @brief During backward propagation, we compute the gradients of the loss with
     * respect to pre-activation input (A), the weights W and bias b. Given ∂L/∂Z
     * we can compute ∂L/∂A, ∂L/∂W and ∂L/∂b as follows:
     * ∂L/∂A = ∂L/∂Z * W
     * ∂L/∂W = (∂L/∂Z)^T * A
     * ∂L/∂b = (∂L/∂Z)^T * ι_N
     *
     * @param dLdZ The gradient of the loss with respect to the pre-activation input
     * @return Eigen::MatrixXd The gradient of the loss with respect to the input A
     */
    Eigen::MatrixXd backward(const Eigen::MatrixXd& dLdZ) override;
};

#endif // LAYER_H
