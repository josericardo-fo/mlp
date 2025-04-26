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
        eps = 1e-12
        y_clp = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true * np.log(y_clp), axis=1))
