from typing import Dict, List, Optional, Tuple
import numpy as np
from src.models.mlp import MLP

class Trainer:
    def __init__(
        self,
        model: MLP,
        batch_size: int = 32,
        epochs: int = 10,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_data = validation_data

    def train(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = True
    ) -> Dict[str, List[float]]:
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        training_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            epoch_acc = 0

            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                batch_loss, batch_acc = self.model.train_batch(X_batch, y_batch)
                epoch_loss += batch_loss * (end_idx - start_idx)
                epoch_acc += batch_acc * (end_idx - start_idx)

            epoch_loss /= n_samples
            epoch_acc /= n_samples

            training_history["loss"].append(epoch_loss)
            training_history["accuracy"].append(epoch_acc)

            if self.validation_data is not None:
                val_loss, val_acc = self.evaluate(*self.validation_data)
                training_history["val_loss"].append(val_loss)
                training_history["val_accuracy"].append(val_acc)

            if verbose:
                val_str = (
                    f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    if self.validation_data is not None
                    else ""
                )
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}{val_str}"
                )

        return training_history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        predictions = self.model.predict(X)
        loss = self.model.loss_function.cross_entropy(y, predictions)
        accuracy = self.model.metrics.calculate_accuracy(y, predictions)
        return loss, accuracy

    def save_model(self, filepath: str) -> None:
        self.model.save(filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "Trainer":
        model = MLP.load(filepath)
        return cls(model)
