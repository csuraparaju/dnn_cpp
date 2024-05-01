/**
 * @file loss.h
 * @author Krish Suraparaju (csurapar)
 * @brief Loss functions are used to quantify the difference between the model's prediction
 * and the actual output. The loss function is a measure of how well the model is
 * performing. The goal of training a neural network is to minimize the loss function.
 *
 * @version 0.1
 * @date 2024-05-01
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>

// Base LossFunction class
class LossFunction {
public:

    /**
     * @brief Stores the model prediction, and other useful variables.
     *
     */
    Eigen::MatrixXd A;  // Model prediction
    Eigen::MatrixXd Y;  // Actual output
    size_t N;  // Number of samples
    size_t C;  // Number of classes
    /**
     * @brief Construct a new Loss Function object, and initialize the
     *        matricies to be empty
     *
     */
    LossFunction() {
        this->N = 0;
        this->C = 0;
        this->A = Eigen::MatrixXd(0, 0);
        this->Y = Eigen::MatrixXd(0, 0);
    }

    /**
     * @brief Destroy the Loss Function object
     *
     */
    virtual ~LossFunction() {}

    /**
     * @brief Computes the loss function given the predicted output and the actual output.
     *
     * @param A The predicted output
     * @param Y The actual output
     * @return double The loss value
     */
    virtual double forward(const Eigen::MatrixXd& A, const Eigen::MatrixXd& Y) = 0;

    /**
     * @brief Computes the derivative of the loss function with respect to the input.
     *
     * @param Y_hat The predicted output
     * @param Y The actual output
     * @return Eigen::MatrixXd The derivative of the loss with respect to the input
     */
    virtual Eigen::MatrixXd backward() = 0;
};

// Concrete class for the Mean Squared Error loss function
class MeanSquaredError : public LossFunction {
public:
    double forward(const Eigen::MatrixXd& A, const Eigen::MatrixXd& Y) override;
    Eigen::MatrixXd backward() override;
};

// Concrete class for the Cross Entropy loss function
class CrossEntropy : public LossFunction {
public:
    double forward(const Eigen::MatrixXd& A, const Eigen::MatrixXd& Y) override;
    Eigen::MatrixXd backward() override;
};

#endif  // LOSS_H