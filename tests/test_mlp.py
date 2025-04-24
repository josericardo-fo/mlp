import unittest

import numpy as np

from src.models.mlp import MLP


class TestMLP(unittest.TestCase):
    def setUp(self):
        self.input_size = 784
        self.hidden_size = 128
        self.output_size = 10
        self.mlp = MLP(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )

    def test_initialization(self):
        self.assertEqual(self.mlp.input_size, self.input_size)
        self.assertEqual(self.mlp.hidden_size, self.hidden_size)
        self.assertEqual(self.mlp.output_size, self.output_size)
        self.assertIsNotNone(self.mlp.weights_input_hidden)
        self.assertIsNotNone(self.mlp.weights_hidden_output)

    def test_forward_pass(self):
        X = np.random.rand(1, self.input_size)
        output = self.mlp.forward(X)
        self.assertEqual(output.shape, (1, self.output_size))

    def test_backward_pass(self):
        X = np.random.rand(1, self.input_size)
        y = np.zeros((1, self.output_size))
        y[0, np.random.randint(0, self.output_size)] = 1  # One-hot encoding
        output = self.mlp.forward(X)
        self.mlp.backward(X, y, output)
        self.assertIsNotNone(self.mlp.weights_input_hidden)
        self.assertIsNotNone(self.mlp.weights_hidden_output)

    def test_loss_calculation(self):
        y_true = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        y_pred = np.array([[0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
        loss = self.mlp.calculate_loss(y_true, y_pred)
        self.assertIsInstance(loss, float)

    def test_accuracy_calculation(self):
        y_true = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        y_pred = np.array([[0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
        accuracy = self.mlp.calculate_accuracy(y_true, y_pred)
        self.assertEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
