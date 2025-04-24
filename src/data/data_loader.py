from typing import Tuple

import numpy as np
import pandas as pd


class DataLoader:
    """Class for loading and preprocessing datasets."""

    @staticmethod
    def load_fashion_mnist(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the Fashion MNIST dataset from a specific file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Tuple containing (X, y) where y is one-hot encoded
        """
        df = pd.read_csv(file_path)

        # Separate features and labels
        y = df["label"].values
        X = df.drop("label", axis=1).values / 255.0  # Normalize the data

        # Convert to one-hot encoding
        y_onehot = np.zeros((y.size, 10))
        y_onehot[np.arange(y.size), y] = 1

        return X, y_onehot
