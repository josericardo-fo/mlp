import numpy as np


class Loss:
    """Class to encapsulate various loss functions."""

    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the cross-entropy loss.

        Args:
            y_true: True labels (one-hot encoded).
            y_pred: Predicted probabilities.

        Returns:
            The cross-entropy loss value.
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
