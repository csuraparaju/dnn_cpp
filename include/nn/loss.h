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
    Eigen::MatrixXd A_;  // Model prediction
    Eigen::MatrixXd Y_;  // Actual output
    size_t N_;  // Number of samples
    size_t C_;  // Number of classes
    /**
     * @brief Construct a new Loss Function object, and initialize the
     *        matricies to be empty
     *
     */
    LossFunction() {
        this->N_ = 0;
        this->C_ = 0;
        this->A_ = Eigen::MatrixXd(0, 0);
        this->Y_ = Eigen::MatrixXd(0, 0);
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


class SoftmaxCrossEntropy : public LossFunction {
public:
    /**
     * @brief Compute the row-wise softmax of a matrix A. Used to
     * transform the model's output into a probability distribution.
     *
     * @param A
     * @return Eigen::MatrixXd
     */
    Eigen::MatrixXd softmax(const Eigen::MatrixXd& A) {
        Eigen::MatrixXd softmax = Eigen::MatrixXd::Zero(A.rows(), A.cols());
        for (int i = 0; i < A.rows(); i++) {
            double max = A.row(i).maxCoeff();
            Eigen::VectorXd exps = (A.row(i).array() - max).exp();
            softmax.row(i) = exps / exps.sum();
        }
        return softmax;
    };
    double forward(const Eigen::MatrixXd& A, const Eigen::MatrixXd& Y) override;
    Eigen::MatrixXd backward() override;
};

#endif  // LOSS_H