# Implementing a Simple Neural Network from Scratch using Python

## Overview
This project demonstrates how to implement a simple neural network from scratch using Python. The implementation includes forward and backward propagation, weight updates using gradient descent, and performance evaluation.

## Features
- Calculation of initial weights using TensorFlow for a single-neuron neural network
- Implementation of a single-neuron neural network from scratch
- Forward propagation and activation functions
- Backpropagation and gradient descent optimization
- Performance evaluation using loss functions
- Comparison with TensorFlow/Keras implementation

## Prerequisites
Before running the project, ensure you have the following dependencies installed:

```bash
pip install numpy matplotlib tensorflow
```

## Project Structure
- `SimpleNN.ipynb` - Jupyter Notebook containing the full implementation and explanation.

## Implementation Details
1. **Weight Calculation:** Initial weights are computed using TensorFlow for a single neuron.
2. **Data Preparation:** Synthetic dataset is used for training.
3. **Network Architecture:** A simple neural network with input, hidden, and output layers.
4. **Activation Functions:** Implementation of ReLU and Sigmoid functions.
5. **Training:** Forward propagation, loss calculation, and backpropagation.
6. **Optimization:** Gradient descent is used to update weights.
7. **Evaluation:** Performance is measured using loss and accuracy.
8. **Comparison:** The custom implementation is validated against TensorFlow/Keras.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/dheru94/implementing-simple-neural-network-with-python.git
   ```
2. Navigate to the project directory:
   ```bash
   cd implementing-simple-neural-network-with-python
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook SimpleNN.ipynb
   ```
4. Run the cells sequentially to see the implementation in action.

## Results
- The neural network successfully learns patterns from the dataset.
- Training loss decreases over iterations, showing learning progress.
- Model performance is comparable to TensorFlow/Keras for a simple NN.

## Future Enhancements
- Implement additional activation functions (e.g., Tanh, Leaky ReLU)
- Extend to deep neural networks with multiple hidden layers
- Use stochastic gradient descent (SGD) and Adam optimizer
- Implement regularization techniques to prevent overfitting

## Contributing
Feel free to fork this repository, make improvements, and submit a pull request.

## License
This project is licensed under the MIT License.

