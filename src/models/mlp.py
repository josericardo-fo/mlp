from typing import Dict, List, Optional, Tuple
import numpy as np
import pickle
import time
from src.utils.activations import Activation
from src.utils.losses import Losses
from src.utils.metrics import Metrics

class MLP:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: str = "sigmoid",
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        use_bias: bool = True,
        weight_init: str = "random",
        random_state: Optional[int] = None,
    ):
        if random_state is not None:
            np.random.seed(random_state)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.use_bias = use_bias
        self.activation_name = activation
        self.activation_func, self.activation_derivative = Activation.get_activation_and_derivative(activation)
        self.loss_function = Losses()
        self.metrics = Metrics()

        self._initialize_weights(weight_init)
        self._initialize_momentum_terms()
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def _initialize_weights(self, method: str) -> None:
        if method == "random":
            self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
            self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
        elif method == "xavier":
            self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1.0 / self.input_size)
            self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1.0 / self.hidden_size)
        elif method == "he":
            self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
            self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        else:
            raise ValueError(f"Unsupported weight initialization method: {method}")

        if self.use_bias:
            self.bias_hidden = np.zeros((1, self.hidden_size))
            self.bias_output = np.zeros((1, self.output_size))

    def _initialize_momentum_terms(self) -> None:
        self.prev_delta_weights_input_hidden = np.zeros((self.input_size, self.hidden_size))
        self.prev_delta_weights_hidden_output = np.zeros((self.hidden_size, self.output_size))
        self.prev_delta_bias_hidden = np.zeros((1, self.hidden_size))
        self.prev_delta_bias_output = np.zeros((1, self.output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        if self.use_bias:
            self.hidden_input += self.bias_hidden
        self.hidden_output = self.activation_func(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        if self.use_bias:
            self.final_input += self.bias_output
        self.final_output = Activation.softmax(self.final_input)  # Usando a função softmax diretamente

        return self.final_output

    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        output_error = output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.activation_derivative(self.hidden_input)

        delta_weights_hidden_output = np.dot(self.hidden_output.T, output_error)
        delta_weights_input_hidden = np.dot(X.T, hidden_error)

        self.weights_hidden_output -= (self.learning_rate * delta_weights_hidden_output + self.momentum * self.prev_delta_weights_hidden_output)
        self.weights_input_hidden -= (self.learning_rate * delta_weights_input_hidden + self.momentum * self.prev_delta_weights_input_hidden)

        self.prev_delta_weights_hidden_output = delta_weights_hidden_output
        self.prev_delta_weights_input_hidden = delta_weights_input_hidden

        if self.use_bias:
            delta_bias_output = np.sum(output_error, axis=0, keepdims=True)
            delta_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

            self.bias_output -= (self.learning_rate * delta_bias_output + self.momentum * self.prev_delta_bias_output)
            self.bias_hidden -= (self.learning_rate * delta_bias_hidden + self.momentum * self.prev_delta_bias_hidden)

            self.prev_delta_bias_output = delta_bias_output
            self.prev_delta_bias_hidden = delta_bias_hidden

    def train_batch(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        output = self.forward(X)
        self.backward(X, y, output)

        loss = self.loss_function.cross_entropy(y, output)
        accuracy = self.metrics.calculate_accuracy(y, output)

        return loss, accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza a predição com o modelo treinado."""
        return self.forward(X)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 32, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, verbose: bool = True) -> Dict[str, List[float]]:
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            epoch_acc = 0

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                batch_loss, batch_acc = self.train_batch(X_batch, y_batch)
                epoch_loss += batch_loss * (end_idx - start_idx)
                epoch_acc += batch_acc * (end_idx - start_idx)

            epoch_loss /= n_samples
            epoch_acc /= n_samples

            if validation_data is not None:
                val_loss, val_acc = self.evaluate(*validation_data)
            else:
                val_loss, val_acc = None, None

            self.training_history["loss"].append(epoch_loss)
            self.training_history["accuracy"].append(epoch_acc)
            if validation_data is not None:
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_accuracy"].append(val_acc)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        return self.training_history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        predictions = self.predict(X)  # Use self.predict em vez de self.model.predict
        loss = self.loss_function.cross_entropy(y, predictions)
        accuracy = self.metrics.calculate_accuracy(y, predictions)
        return loss, accuracy

    def save(self, filepath: str) -> None:
        model_data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "weights_input_hidden": self.weights_input_hidden,
            "weights_hidden_output": self.weights_hidden_output,
            "bias_hidden": self.bias_hidden if self.use_bias else None,
            "bias_output": self.bias_output if self.use_bias else None,
            "activation": self.activation_name if hasattr(self, 'activation_name') else "sigmoid",  # Valor padrão
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "use_bias": self.use_bias,
            "training_history": self.training_history,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str) -> "MLP":
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        model = cls(
            input_size=model_data["input_size"],
            hidden_size=model_data["hidden_size"],
            output_size=model_data["output_size"],
            activation=model_data["activation"],
            learning_rate=model_data["learning_rate"],
            momentum=model_data["momentum"],
            use_bias=model_data["use_bias"],
        )

        model.weights_input_hidden = model_data["weights_input_hidden"]
        model.weights_hidden_output = model_data["weights_hidden_output"]

        if model.use_bias:
            model.bias_hidden = model_data["bias_hidden"]
            model.bias_output = model_data["bias_output"]

        model.training_history = model_data["training_history"]

        return model