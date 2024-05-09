#include "../../include/nn/layer.h"
#include <Eigen/Dense>
#include <iostream>

Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd& A) {
    size_t N = A.rows();
    this->A_ = A;
    this->N_ = A.rows();
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(this->N_, 1);
    Eigen::MatrixXd W = this->W_;
    Eigen::MatrixXd b = this->b_;
    Eigen::MatrixXd Z = A * W.transpose() + one * b.transpose();
    return Z;
}

Eigen::MatrixXd Linear::backward(const Eigen::MatrixXd& dLdZ) {
    Eigen::MatrixXd dLdA = dLdZ * this->W_;
    Eigen::MatrixXd one = Eigen::MatrixXd::Ones(this->N_, 1);
    this->dLdW_ = dLdZ.transpose() * this->A_;
    this->dLdb_ = dLdZ.transpose() * one;
    return dLdA;
}

