#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/nn/loss.h"
#include <iostream>

using namespace Eigen;

TEST(MSETest, Forward) {
    MeanSquaredError mse;
    MatrixXd A(4, 2);
    A << -4.0, -3.0,
         -2.0, -1.0,
          0.0,  1.0,
          2.0,  3.0;

    MatrixXd Y(4, 2);
    Y << 0.0, 1.0,
         1.0, 0.0,
         1.0, 0.0,
         0.0, 1.0;


    double loss = mse.forward(A, Y);

    EXPECT_NEAR(loss, 6.5, 1e-8);
}

TEST(MSETest, Backward) {
    MeanSquaredError mse;
    MatrixXd A(4, 2);
    A << -4.0, -3.0,
         -2.0, -1.0,
          0.0,  1.0,
          2.0,  3.0;

    MatrixXd Y(4, 2);
    Y << 0.0, 1.0,
         1.0, 0.0,
         1.0, 0.0,
         0.0, 1.0;

    double loss = mse.forward(A, Y);
    MatrixXd dLdA = mse.backward();
    MatrixXd expected_dLdA(4, 2);
    expected_dLdA << -1.0, -1.0,
                     -0.75, -0.25,
                     -0.25,  0.25,
                      0.5,   0.5;

    EXPECT_TRUE(dLdA.isApprox(expected_dLdA, 1e-8));
}

TEST(CrossEntropyTest, Forward) {
    SoftmaxCrossEntropy cross_entropy;
    MatrixXd A(4, 2);
    A << -4, -3,
         -2, -1,
          0,  1,
          2,  3;
    MatrixXd Y(4, 2);
    Y << 0, 1,
         1, 0,
         1, 0,
         0, 1;
    double loss = cross_entropy.forward(A, Y);
    double expected_loss = 0.8133;
    ASSERT_NEAR(loss, expected_loss, 1e-4);
}

TEST(CrossEntropyTest, Backward) {
    SoftmaxCrossEntropy cross_entropy;
    MatrixXd A(4, 2);
    A << -4, -3,
         -2, -1,
          0,  1,
          2,  3;
    MatrixXd Y(4, 2);
    Y << 0, 1,
         1, 0,
         1, 0,
         0, 1;
    cross_entropy.forward(A, Y);
    MatrixXd dLdA = cross_entropy.backward();
    MatrixXd expected_dLdA(4, 2);
    expected_dLdA << 0.067, -0.067,
                     -0.183,  0.183,
                     -0.183,  0.183,
                      0.067, -0.067;

    ASSERT_TRUE(dLdA.isApprox(expected_dLdA, 1e-2));
}
