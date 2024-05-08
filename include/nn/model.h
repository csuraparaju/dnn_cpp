/**
 * @file model.h
 * @author Krish Suraparaju (csurapar@andrew.cmu.edu)
 * @brief We can think of a neural network (NN) as a mathematical function
 * which takes an input data x and computes an output y: y = fNN (x)
 * The function fNN is a nested function, where each sub function
 * can be thought of as a layer in the neural network. The output of
 * one layer is passed as input to the next layer, and so on, until
 * we reach the final layer which produces the output y. That is,
 * y = fNN (x) = fL (fL-1 ( ... f2 (f1 (x)) ... )) where f1, f2, ... fL
 * are vector valued functions of the form fi (x) = g_i(W_i * x + b_i).
 * Here, W_i is the weight matrix, b_i is the bias vector, and g_i is the
 * activation function applied element-wise to the input. The parameters
 * W_i and b_i are learned during the training process using an optimization
 * algorithm such as gradient descent.
 *
 * @version 0.1
 * @date 2024-05-08
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <utility>
#include <vector>
#include "layer.h"
#include "loss.h"
#include "activation.h"

// Base class for all neural network models
class Model {
public:

    /**
     * @brief Fields that every model should have.
     * Store unique_ptr to layers, loss function, and activation functions.
     * This is to ensure that the memory is managed properly.
     */
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<LossFunction> loss;
    std::vector<std::unique_ptr<ActivationFunction>> activations;

    /**
     * @brief Construct a new Model object. Take in a vector of
     *       layers, a loss function, and a vector of activation
     *       functions, and store them in the model.
     *
     */
    Model(std::vector<std::unique_ptr<Layer>> layers,
          std::unique_ptr<LossFunction> loss,
          std::vector<std::unique_ptr<ActivationFunction>> activations) {
        this->layers = layers;
        this->loss = loss;
        this->activations = activations;
    }

    /**
     * @brief Forward pass of the model
     *
     * @param X The input data
     * @return Eigen::MatrixXd The output of the model
     */
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& X) = 0;

    /**
     * @brief Backward pass of the model
     *
     * @param dLdZ The gradient of the loss with respect to the output of the model
     * @return Eigen::MatrixXd The gradient of the loss with respect to the input of the model
     */
    virtual Eigen::MatrixXd backward() = 0;


    /**
     * @brief Destroy the Model object
     *
     */
    virtual ~Model() {}
};

// Concrete class for a feedforward neural network model
class LinearModel : public Model {
public:

    /**
     * @brief Construct a new Linear Model object. Takes in
     * the input size, output size, num linear layers, loss
     * function, and the activation functions to use.
     *
     */
    LinearModel(size_t input_size, size_t output_size, size_t num_layers,
                std::unique_ptr<LossFunction> loss,
                std::vector<std::unique_ptr<ActivationFunction>> activations) :
                Model({}, std::move(loss), std::move(activations)) {
        for(size_t i = 0; i < num_layers; i++) {
            if(i == 0) {
                layers.push_back(std::make_unique<Linear>(input_size, output_size));
            } else {
                layers.push_back(std::make_unique<Linear>(output_size, output_size));
            }
        }
    }

    /**
     * @brief Forward pass of the model
     *
     * @param X The input data
     * @return Eigen::MatrixXd The output of the model
     */
    Eigen::MatrixXd forward(const Eigen::MatrixXd& X) override {
        Eigen::MatrixXd A = X;
        for(size_t i = 0; i < layers.size(); i++) {
            A = layers[i]->forward(A);
            A = activations[i]->forward(A);
        }
        return A;
    }

    /**
     * @brief Backward pass of the model
     *
     * @param dLdZ The gradient of the loss with respect to the output of the model
     * @return Eigen::MatrixXd The gradient of the loss with respect to the input of the model
     */
    Eigen::MatrixXd backward() override {
        Eigen::MatrixXd dLdZ = loss->backward();
        for(int i = layers.size() - 1; i >= 0; i--) {
            dLdZ = activations[i]->backward(dLdZ);
            dLdZ = layers[i]->backward(dLdZ);
        }
        return dLdZ;
    }
};

#endif