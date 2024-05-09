#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/nn/layer.h"
#include <iostream>

TEST(LinearTest, Forward) {
    // Will initially be random, but we set it to a known value for testing
    Linear linear(2, 3);
    linear.W_ = Eigen::MatrixXd(3, 2);
    linear.W_ << -2.0, -1.0,
                 0.0,  1.0,
                 2.0,  3.0;
    linear.b_ = Eigen::MatrixXd(3, 1);
    linear.b_ << -1.0,
                 0.0,
                 1.0;

    Eigen::MatrixXd A(4, 2);
    A << -4.0, -3.0,
         -2.0, -1.0,
          0.0,  1.0,
          2.0,  3.0;

    Eigen::MatrixXd Z = linear.forward(A);


    Eigen::MatrixXd expected_Z(4, 3);
    expected_Z << 10.0, -3.0, -16.0,
                   4.0, -1.0,  -6.0,
                  -2.0,  1.0,   4.0,
                  -8.0,  3.0,  14.0;

    ASSERT_TRUE(Z.isApprox(expected_Z, 1e-12));
}

TEST(LinearTest, Backward) {
    Linear linear(2, 3);
    linear.W_ = Eigen::MatrixXd(3, 2);
    linear.W_ << -2.0, -1.0,
                 0.0,  1.0,
                 2.0,  3.0;
    linear.b_ = Eigen::MatrixXd(3, 1);
    linear.b_ << -1.0,
                 0.0,
                 1.0;

    Eigen::MatrixXd A(4, 2);
    A << -4.0, -3.0,
         -2.0, -1.0,
          0.0,  1.0,
          2.0,  3.0;

    linear.forward(A);

    Eigen::MatrixXd dLdZ(4, 3);
    dLdZ << -4.0, -3.0, -2.0,
            -1.0, 0.0, 1.0,
            2.0, 3.0, 4.0,
            5.0, 6.0, 7.0;

    Eigen::MatrixXd dLdA = linear.backward(dLdZ);

    Eigen::MatrixXd expected_dLdA(4, 2);
    expected_dLdA << 4.0, -5.0,
                      4.0,  4.0,
                      4.0, 13.0,
                      4.0, 22.0;

    ASSERT_TRUE(dLdA.isApprox(expected_dLdA, 1e-12));

    Eigen::MatrixXd expected_dLdW(3, 2);
    expected_dLdW << 28.0, 30.0,
                     24.0, 30.0,
                     20.0, 30.0;

    Eigen::MatrixXd expected_dLdb(3, 1);
    expected_dLdb << 2.0,
                     6.0,
                    10.0;

    ASSERT_TRUE(linear.dLdW_.isApprox(expected_dLdW, 1e-12));
    ASSERT_TRUE(linear.dLdb_.isApprox(expected_dLdb, 1e-12));
}

