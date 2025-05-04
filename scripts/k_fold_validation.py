import os
import sys
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_loader import DataLoader
from src.models.mlp import MLP
from src.trainers.trainer import Trainer

def k_fold_validation(data_path, k=5):
    print("Carregando dados para K-Fold Validation...")
    X, y = DataLoader.load_fashion_mnist(data_path)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold = 1
    best_model = None
    best_accuracy = 0
    best_fold = None
    all_test_accuracies = []
    all_test_losses = []

    for train_index, test_index in kf.split(X):
        print(f"\n--- Fold {fold}/{k} ---")

        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

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
            validation_data=(X_val, y_val),
        )

        print(f"Treinando o modelo no Fold {fold}...")
        history = trainer.train(X_train, y_train)

        print(f"Avaliando o modelo no Fold {fold}...")
        val_loss, val_acc = trainer.evaluate(X_val, y_val)
        print(f"Fold {fold} - Loss no conjunto de validação: {val_loss:.4f}")
        print(f"Fold {fold} - Acurácia no conjunto de validação: {val_acc:.4f}")

        all_test_losses.append(val_loss)
        all_test_accuracies.append(val_acc)

        if history and "val_loss" in history and "val_accuracy" in history:
            plt.figure(figsize=(12, 5))

            # Gráfico de perda
            plt.subplot(1, 2, 1)
            plt.plot(history["loss"], label="Loss de Treinamento")
            plt.plot(history["val_loss"], label="Loss de Validação")
            plt.title(f"Fold {fold} - Loss ao longo das épocas")
            plt.xlabel("Épocas")
            plt.ylabel("Loss")
            plt.legend()

            # Gráfico de acurácia
            plt.subplot(1, 2, 2)
            plt.plot(history["accuracy"], label="Acurácia de Treinamento")
            plt.plot(history["val_accuracy"], label="Acurácia de Validação")
            plt.title(f"Fold {fold} - Acurácia ao longo das épocas")
            plt.xlabel("Épocas")
            plt.ylabel("Acurácia")
            plt.legend()

            plt.tight_layout()
            plt.show()

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = model
            best_fold = fold

        fold += 1

    print("\n--- Resultados Finais do K-Fold ---")
    print(f"Média da Loss nos folds: {np.mean(all_test_losses):.4f} ± {np.std(all_test_losses):.4f}")
    print(f"Média da Acurácia nos folds: {np.mean(all_test_accuracies):.4f} ± {np.std(all_test_accuracies):.4f}")
    print(f"Melhor Fold: {best_fold} com Acurácia: {best_accuracy:.4f}")

    return best_model, best_fold

def evaluate_on_test_set(model, test_data_path):
    print("\n--- Avaliação no Conjunto de Teste ---")
    X_test, y_test = DataLoader.load_fashion_mnist(test_data_path)

    trainer = Trainer(
        model=model,
        batch_size=32,
        epochs=0,  
    )

    test_loss, test_acc = trainer.evaluate(X_test, y_test)
    print(f"Loss no conjunto de teste: {test_loss:.4f}")
    print(f"Acurácia no conjunto de teste: {test_acc:.4f}")

    plt.figure(figsize=(6, 4))
    plt.bar(["Loss", "Acurácia"], [test_loss, test_acc], color=["red", "blue"])
    plt.title("Resultados no Conjunto de Teste")
    plt.ylabel("Valor")
    plt.show()

if __name__ == "__main__":
    best_model, best_fold = k_fold_validation("data/fashion-mnist_train.csv", k=5)

    print(f"\n--- Avaliando o Melhor Modelo do Fold {best_fold} no Conjunto de Teste ---")
    evaluate_on_test_set(best_model, "data/fashion-mnist_test.csv")