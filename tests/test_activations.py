import numpy as np


import unittest
from src.utils.activations import Activations

class TestActivations(unittest.TestCase):

    def setUp(self):
        self.activations = Activations()

    def test_sigmoid(self):
        input_data = np.array([-1, 0, 1])
        expected_output = 1 / (1 + np.exp(-input_data))
        np.testing.assert_almost_equal(self.activations.sigmoid(input_data), expected_output)

    def test_sigmoid_derivative(self):
        input_data = np.array([-1, 0, 1])
        sigmoid_output = self.activations.sigmoid(input_data)
        expected_derivative = sigmoid_output * (1 - sigmoid_output)
        np.testing.assert_almost_equal(self.activations.sigmoid_derivative(input_data), expected_derivative)

    def test_relu(self):
        input_data = np.array([-1, 0, 1])
        expected_output = np.maximum(0, input_data)
        np.testing.assert_almost_equal(self.activations.relu(input_data), expected_output)

    def test_relu_derivative(self):
        input_data = np.array([-1, 0, 1])
        expected_derivative = np.where(input_data > 0, 1, 0)
        np.testing.assert_almost_equal(self.activations.relu_derivative(input_data), expected_derivative)

    def test_tanh(self):
        input_data = np.array([-1, 0, 1])
        expected_output = np.tanh(input_data)
        np.testing.assert_almost_equal(self.activations.tanh(input_data), expected_output)

    def test_tanh_derivative(self):
        input_data = np.array([-1, 0, 1])
        expected_derivative = 1 - np.tanh(input_data) ** 2
        np.testing.assert_almost_equal(self.activations.tanh_derivative(input_data), expected_derivative)

if __name__ == '__main__':
    unittest.main()