#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../include/nn/activation.h"

TEST(ReLUForwardTest, ForwardTest) {
    ReLU relu = ReLU();
    Eigen::MatrixXd Z(2, 3);
    Z << 0.0378, 0.3022, -1.6123,
         -2.5186, -1.9395, 1.4077;
    Eigen::MatrixXd A = relu.forward(Z);
    Eigen::MatrixXd expected(2, 3);
    expected << 0.0378, 0.3022, 0.0,
                0.0, 0.0, 1.4077;
    ASSERT_TRUE(A.isApprox(expected, 1e-12));
}

TEST(ReLUBackwardTest, BackwardTest) {
    ReLU relu;
    Eigen::MatrixXd Z(2, 3);
    Z << 0.0378, 0.3022, -1.6123,
         -2.5186, -1.9395, 1.4077;
    relu.forward(Z);
    Eigen::MatrixXd dLdA(2, 3);
    dLdA << 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0;
    Eigen::MatrixXd dLdZ = relu.backward(dLdA);
    Eigen::MatrixXd expected(2, 3);
    expected << 1.0, 2.0, 0.0,
                0.0, 0.0, 6.0;
    ASSERT_TRUE(dLdZ.isApprox(expected, 1e-12));
}

TEST(SigmoidForwardTest, ForwardTest) {
    Sigmoid sigmoid; // Assuming Sigmoid is the class name for your activation function
    Eigen::MatrixXd Z(4, 2);
    Z << -4.0, -3.0,
         -2.0, -1.0,
          0.0,  1.0,
          2.0,  3.0;
    Eigen::MatrixXd A = sigmoid.forward(Z);
    Eigen::MatrixXd expected(4, 2);
    expected << 0.018, 0.0474,
                0.1192, 0.2689,
                0.5,   0.7311,
                0.8808, 0.9526;
    ASSERT_TRUE(A.isApprox(expected, 1e-3));
}

TEST(SigmoidBackwardTest, BackwardTest) {
    Sigmoid sigmoid;
    Eigen::MatrixXd Z(4, 2);
    Z << -4.0, -3.0,
         -2.0, -1.0,
          0.0,  1.0,
          2.0,  3.0;
    sigmoid.forward(Z);
    Eigen::MatrixXd dLdA(4, 2);
    dLdA << 1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0;
    Eigen::MatrixXd dLdZ = sigmoid.backward(dLdA);
    Eigen::MatrixXd expected(4, 2);

    expected << 0.0177, 0.0452,
                0.105,  0.1966,
                0.25,   0.1966,
                0.105,  0.0452;
    ASSERT_TRUE(dLdZ.isApprox(expected, 1e-3));
}

TEST(TanhForwardTest, ForwardTest) {
    Tanh tanh; // Assuming Tanh is the class name for your activation function
    Eigen::MatrixXd Z(4, 2);
    Z << -4.0, -3.0,
         -2.0, -1.0,
          0.0,  1.0,
          2.0,  3.0;
    Eigen::MatrixXd A = tanh.forward(Z);
    Eigen::MatrixXd expected(4, 2);
    expected << -0.9993, -0.9951,
                -0.964,  -0.7616,
                 0.0,     0.7616,
                 0.964,   0.9951;
    ASSERT_TRUE(A.isApprox(expected, 1e-3));
}

TEST(TanhBackwardTest, BackwardTest) {
    Tanh tanh;
    Eigen::MatrixXd Z(4, 2);
    Z << -4.0, -3.0,
         -2.0, -1.0,
          0.0,  1.0,
          2.0,  3.0;
    tanh.forward(Z);
    Eigen::MatrixXd dLdA(4, 2);
    dLdA << 1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0;
    Eigen::MatrixXd dLdZ = tanh.backward(dLdA);
    Eigen::MatrixXd expected(4, 2);
    expected << 0.0013, 0.0099,
                0.0707, 0.42,
                1.0,    0.42,
                0.0707, 0.0099;
    ASSERT_TRUE(dLdZ.isApprox(expected, 1e-3));
}
