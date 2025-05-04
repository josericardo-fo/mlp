import os
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_loader import DataLoader
from src.models.mlp import MLP
from src.trainers.trainer import Trainer

def train_and_evaluate(data_path, description):
    print(f"Carregando dados ({description})...")
    X, y = DataLoader.load_fashion_mnist(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Inicializando modelo ({description})...")
    model = MLP(
        input_size=784,
        hidden_size=128,
        output_size=10,
        activation="relu",
        learning_rate=0.01,
        weight_init="he",
    )

    trainer = Trainer(
        model=model,
        batch_size=32,
        epochs=20,
        validation_data=(X_test, y_test),
    )

    print(f"Treinando o modelo ({description})...")
    history = trainer.train(X_train, y_train)

    if history and "val_loss" in history and "val_accuracy" in history:
        plt.figure(figsize=(12, 5))

        # Gráfico de perda
        plt.subplot(1, 2, 1)
        plt.plot(history["loss"], label="Loss de Treinamento")
        plt.plot(history["val_loss"], label="Loss de Validação")
        plt.title(f"Loss ao longo das epoch ({description})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Gráfico de acurácia
        plt.subplot(1, 2, 2)
        plt.plot(history["accuracy"], label="Acurácia de Treinamento")
        plt.plot(history["val_accuracy"], label="Acurácia de Validação")
        plt.title(f"Acurácia ao longo das epoch ({description})")
        plt.xlabel("Epoch")
        plt.ylabel("Acurácia")
        plt.legend()

        plt.tight_layout()
        plt.show()

    print(f"Avaliando o modelo ({description})...")
    test_loss, test_acc = trainer.evaluate(X_test, y_test)
    print(f"Loss no conjunto de teste ({description}): {test_loss:.4f}")
    print(f"Acurácia no conjunto de teste ({description}): {test_acc:.4f}")

def holdout_validation():
    # Treino com o arquivo train.csv
    train_and_evaluate("data/fashion-mnist_train.csv", "Treinamento com train.csv")

    # Treino com o arquivo test.csv
    train_and_evaluate("data/fashion-mnist_test.csv", "Treinamento com test.csv")

if __name__ == "__main__":
    holdout_validation()