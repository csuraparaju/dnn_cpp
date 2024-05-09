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
    if(this->A.size() == 0 || this->Y.size() == 0
        || this->N == 0 || this->C == 0) {
        this->A = A;
        this->Y = Y;
        this->N = A.rows();
        this->C = A.cols();
    }
    double loss = (A - Y).array().square().sum();
    return loss / (N * C);

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
    Eigen::MatrixXd A = this->A;
    Eigen::MatrixXd Y = this->Y;
    return 2 * (A - Y) / (N * C);
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
    if(this->A.size() == 0 || this->Y.size() == 0
        || this->N == 0 || this->C == 0) {
        this->A = A;
        this->Y = Y;
        this->N = A.rows();
        this->C = A.cols();
    }

    Eigen::MatrixXd softmax = SoftmaxCrossEntropy::softmax(A);

    Eigen::MatrixXd log_softmax = softmax.array().log();
    double loss = -1 * (Y.cwiseProduct(log_softmax)).sum();
    double mean_loss = loss / N;

    return mean_loss;
}


/**
 * @brief Computes the derivative of the cross-entropy loss function with respect
 *        to the input.
 *
 * @return Eigen::MatrixXd The derivative of the loss with respect to the input
 */
Eigen::MatrixXd SoftmaxCrossEntropy::backward() {
    Eigen::MatrixXd A = this->A;
    Eigen::MatrixXd Y = this->Y;
    Eigen::MatrixXd softmax = SoftmaxCrossEntropy::softmax(A);
    return (softmax - Y) / N;
}