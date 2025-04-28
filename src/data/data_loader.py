from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataLoader:
    """Classe para carregar e processar os dados."""

    def __init__(self, batch_size: int = 32, shuffle: bool = True):
        """
        Inicializa o DataLoader com o tamanho do mini-batch e se os dados devem ser embaralhados.

        Args:
            batch_size: tamanho do mini-batch
            shuffle: se os dados devem ser embaralhados
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.classes = [
            'T-shirt/top', 'Calça', 'Pullover', 'Vestido', 'Casaco',
            'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota'
        ]

    @staticmethod
    def load_fashion_mnist(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega o dataset Fashion MNIST de um arquivo específico.

        Args:
            file_path: Caminho para o arquivo CSV

        Returns:
            Tupla contendo (X, y) onde y está em one-hot encoding
        """
        df = pd.read_csv(file_path)

        # Separa features e labels
        y = df["label"].values
        X = df.drop("label", axis=1).values.reshape(-1, 28, 28)

        # Converte para one-hot encoding
        y_onehot = np.zeros((y.size, 10))
        y_onehot[np.arange(y.size), y] = 1

        return X, y_onehot

    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pré-processa os dados apenas com normalização (0-1).

        Args:
            X: Features de entrada (imagens)
            y: Labels alvo

        Returns:
            Tupla de (X, y) pré-processados
        """
        X_normalized = X / 255.0
        return X_normalized, y

    def visualize_batch(self, X_batch: np.ndarray, y_batch: np.ndarray, n_samples: int = 5):
        """
        Visualiza um batch de imagens com suas labels.

        Args:
            X_batch: Batch de imagens
            y_batch: Batch de labels
            n_samples: Número de amostras para visualizar
        """
        plt.figure(figsize=(15, 3))
        for i in range(min(n_samples, len(X_batch))):
            plt.subplot(1, n_samples, i + 1)
            plt.imshow(X_batch[i], cmap='gray')
            plt.title(self.classes[np.argmax(y_batch[i])])
            plt.axis('off')
        plt.show()

    def create_mini_batches(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Cria mini-batches do dataset.

        Args:
            X: Features de entrada
            y: Labels alvo

        Returns:
            Lista de tuplas (X_batch, y_batch)
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            batches.append((X_batch, y_batch))
        
        return batches

    def benchmark_preprocessing(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compara o desempenho com e sem normalização.

        Args:
            X: Features de entrada
            y: Labels alvo

        Returns:
            Dicionário com resultados do benchmark
        """
        # Com normalização
        X_normalized = X / 255.0
        return {
            "normalized": X_normalized
        }
