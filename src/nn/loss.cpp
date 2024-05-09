/**
 * @file loss.cpp
 * @author Krish Suraparaju (csurapar)
 * @brief Provides the concrete implementation of the various loss
 *       functions defined in include/nn/loss.h header.
 * @version 0.1
 * @date 2024-05-01
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "../../include/nn/loss.h"

/**
 * @brief Computes the mean squared error loss function given the predicted output
 *        and the actual output.
 *
 * @param A The predicted output
 * @param Y The actual output
 * @return double The loss value
 */
double MeanSquaredError::forward(const Eigen::MatrixXd& A, const Eigen::MatrixXd& Y) {

    // If the matrices are empty, initialize them to the input value A and Y
    if(this->A_.size() == 0 || this->Y_.size() == 0
        || this->N_ == 0 || this->C_ == 0) {
        this->A_ = A;
        this->Y_ = Y;
        this->N_ = A.rows();
        this->C_ = A.cols();
    }
    double loss = (A - Y).array().square().sum();
    return loss / (N_ * C_);

}

/**
 * @brief Computes the derivative of the mean squared error loss function with respect
 *        to the input.
 *
 * @param Y_hat The predicted output
 * @param Y The actual output
 * @return Eigen::MatrixXd The derivative of the loss with respect to the input
 */
Eigen::MatrixXd MeanSquaredError::backward() {
    Eigen::MatrixXd A = this->A_;
    Eigen::MatrixXd Y = this->Y_;
    return 2 * (A - Y) / (N_ * C_);
}


/**
 * @brief Computes the cross-entropy loss function given the predicted output
 *        and the actual output.
 *
 * @param A The predicted output
 * @param Y The actual output
 * @return double The loss value
 */
double SoftmaxCrossEntropy::forward(const Eigen::MatrixXd& A, const Eigen::MatrixXd& Y) {
    // If the matrices are empty, initialize them to the input value A and Y
    if(this->A_.size() == 0 || this->Y_.size() == 0
        || this->N_ == 0 || this->C_ == 0) {
        this->A_ = A;
        this->Y_ = Y;
        this->N_ = A.rows();
        this->C_ = A.cols();
    }

    Eigen::MatrixXd softmax = SoftmaxCrossEntropy::softmax(A);

    Eigen::MatrixXd log_softmax = softmax.array().log();
    double loss = -1 * (Y.cwiseProduct(log_softmax)).sum();
    double mean_loss = loss / N_;

    return mean_loss;
}


/**
 * @brief Computes the derivative of the cross-entropy loss function with respect
 *        to the input.
 *
 * @return Eigen::MatrixXd The derivative of the loss with respect to the input
 */
Eigen::MatrixXd SoftmaxCrossEntropy::backward() {
    Eigen::MatrixXd A = this->A_;
    Eigen::MatrixXd Y = this->Y_;
    Eigen::MatrixXd softmax = SoftmaxCrossEntropy::softmax(A);
    return (softmax - Y) / N_;
}