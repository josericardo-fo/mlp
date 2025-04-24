import numpy as np


class Metrics:
    """Classe para calcular métricas de desempenho do modelo."""

    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula a acurácia das predições.

        Args:
            y_true: Rótulos verdadeiros (one-hot encoded).
            y_pred: Rótulos preditos (one-hot encoded).

        Returns:
            Acurácia como um valor entre 0 e 1.
        """
        y_true_class = np.argmax(y_true, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        return np.mean(y_true_class == y_pred_class)

    @staticmethod
    def calculate_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula a perda de entropia cruzada.

        Args:
            y_true: Rótulos verdadeiros (one-hot encoded).
            y_pred: Rótulos preditos (probabilidades).

        Returns:
            Perda como um valor escalar.
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Evitar log(0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gera a matriz de confusão.

        Args:
            y_true: Rótulos verdadeiros (one-hot ou inteiros).
            y_pred: Rótulos preditos (one-hot ou inteiros).

        Returns:
            Matriz de confusão (np.ndarray).
        """
        # Se for one-hot, converte para inteiros
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm
