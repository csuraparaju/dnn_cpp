# Deep Learning Library in C++

This is a deep learning library implemented in C++, providing functionalities for building and training neural networks.
It is organized into different modules, each containing classes and functions for specific features. The organization of the project is inspired by PyTorch, which makes most components into standalone classes.

## Implemented Features

- Activation functions
    - Sigmoid
    - Tanh
    - ReLU
- Loss functions
    - Mean Squared Error
    - Cross-Entropy

## To-Do List

- Implement layer functionalities (e.g., fully connected, convolutional, recurrent)
- Implement loss functions (e.g., mean squared error, cross-entropy)
- Implement model building and training functionalities
- Implement optimization algorithms (e.g., Stochastic Gradient Descent)
- Implement evaluation metrics (e.g., accuracy, precision, recall)
- Add support for various neural network architectures (e.g., CNNs, RNNs, GANs)
- Enhance testing coverage for all implemented functionalities
- Improve documentation for better usability and understanding

## Getting Started

To use this library, you can include the necessary header files from the `include` directory in your C++ project. Make sure to link the compiled library with your project.

For running tests, navigate to the `dnn_tests` directory and compile the test files. Execute the compiled binaries to run the tests.

## Contributions

Contributions to this project are welcome. Feel free to open issues for bugs or feature requests, and submit pull requests with enhancements.

## Project Structure
```
├── dnn_tests
│   ├── activation_test.cpp
│   └── build_test.cpp
├── include
│   ├── nn
│   │   ├── activation.h
│   │   ├── layer.h
│   │   ├── loss.h
│   │   └── model.h
│   └── optim
│       └── sgd.h
├── readme.md
└── src
    ├── nn
    │   ├── activation.cpp
    │   ├── layer.cpp
    │   ├── loss.cpp
    │   └── model.cpp
    └── optim
        └── sgd.cpp
```

## Build Instructions
1. Clone the repository
2. Install googletest and Eigen libraries using homebrew
    ```brew install eigen```and ```brew install googletest```. If you are on windows or linux, read the installation instructions from the respective websites.
3. Run the following commands to build the project
    ```$> mkdir build
       $> cd build
       $> cmake ..
       $> make
    ```
4. Run the tests
    ```.build/dnn_tests/```


## License

This project is licensed under the [MIT License](LICENSE).
