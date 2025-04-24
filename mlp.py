import numpy as np
import pandas as pd
import torch

class MLP:
    def __init__(self, activation="relu", learning_rate=0.01, epochs=1000):
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_size = 28 * 28
        self.hidden_size = 256
        self.output_size = 10
        self.weights = []
        self.biases = []
        self.losses = []
        self.accuracies = []
        self.layers = [self.input_size, self.hidden_size, self.output_size]

    def _load_data(self):
        train = pd.read_csv("data/fashion_train.csv")
        test = pd.read_csv("data/fashion_test.csv")
        x_train = train.drop("label", axis=1).values
        y_train = train["label"].values
        x_test = test.drop("label", axis=1).values
        y_test = test["label"].values
        return x_train, y_train, x_test, y_test

    def _process_data(self, x_train, y_train, x_test, y_test):
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        y_train = pd.get_dummies(y_train).values
        y_test = pd.get_dummies(y_test).values
        return x_train, y_train, x_test, y_test

    def _initialize_xavier(self):
        # Inicialização dos pesos - usando escala 1/sqrt(n) para melhor convergência
        W1 = np.random.randn(self.layers[0], self.layers[1]) * np.sqrt(
            1 / self.layers[0]
        )
        b1 = np.zeros(self.layers[1])  # Bias para a camada oculta
        W2 = np.random.randn(self.layers[1], self.layers[2]) * np.sqrt(
            1 / self.layers[1]
        )
        b2 = np.zeros(self.layers[2])  # Bias para a camada de saída
        self.weights = [W1, W2]
        self.biases = [b1, b2]
        for i in range(len(self.layers) - 1):
            limit = np.sqrt(2 / self.layers[i])
            self.weights.append(
                np.random.randn(self.layers[i], self.layers[i + 1]) * limit
            )
            self.biases.append(np.zeros((1, self.layers[i + 1])))

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _forward_pass(self, x):
        self.z1 = np.dot(x, self.weights[0]) + self.biases[0]
        self.a1 = self._relu(self.z1)
        output = np.dot(self.a1, self.weights[1]) + self.biases[1]
        probs = self._softmax(output)
        return probs

    def _cross_entropy_loss(self, probs, y_true):
        epsilon = 1e-15  # Detalhe de implementação para evitar log de 0 (náo existe)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred), axis=1).mean()
        return loss

    def _backward_pass(self, x, y_true):
        probs = self._foward_pass(x)
        loss = self._cross_entropy_loss(probs, y_true)

        # REVISAR DAQUI PARA BAIXO
        delta_2 = probs - y_true  
        grad_weights2 = np.dot(self.a1.T, delta_2) 
        grad_biases2 = np.sum(delta_2, axis=0, keepdims=True) 

        delta_1 = np.dot(delta_2, self.weights[1].T) * (self.a1 > 0) 
        grad_weights1 = np.dot(x.T, delta_1)  
        grad_biases1 = np.sum(delta_1, axis=0, keepdims=True)  

        self.weights[1] -= self.learning_rate * grad_weights2
        self.biases[1] -= self.learning_rate * grad_biases2
        self.weights[0] -= self.learning_rate * grad_weights1
        self.biases[0] -= self.learning_rate * grad_biases1

        return loss
