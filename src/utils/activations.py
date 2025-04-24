import numpy as np

class Activations:
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of the sigmoid function."""
        sx = Activations.sigmoid(x)
        return sx * (1 - sx)

    @staticmethod
    def tanh(x):
        """Tanh activation function."""
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        """Derivative of the tanh function."""
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        """ReLU activation function."""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """Derivative of the ReLU function."""
        return np.where(x > 0, 1, 0)