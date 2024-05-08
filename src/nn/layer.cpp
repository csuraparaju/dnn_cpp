#include "../../include/nn/layer.h"
#include <Eigen/Dense>
#include <iostream>

Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd& A) {
    size_t N = A.rows();
    this->A = A;
    this->N = N;
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(N, 1);
    Eigen::MatrixXd W = this->W;
    Eigen::MatrixXd b = this->b;
    Eigen::MatrixXd Z = A * W.transpose() + one * b.transpose();
    return Z;
}

Eigen::MatrixXd Linear::backward(const Eigen::MatrixXd& dLdZ) {
    Eigen::MatrixXd dLdA = dLdZ * this->W;
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(this->N, 1);
    this->dLdW = dLdZ.transpose() * this->A;
    this->dLdb = dLdZ.transpose() * one;
    return dLdA;
}

