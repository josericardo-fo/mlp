import os, sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_loader import DataLoader
from src.models.mlp import MLP
from src.trainers.trainer import Trainer
from src.utils.metrics import Metrics

def run_holdout_experiment():
    x_train, y_train = DataLoader.load_fashion_mnist("src/data/fashion_train.csv")
    x_test, y_test = DataLoader.load_fashion_mnist("src/data/fashion_test.csv")

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
        validation_data=(x_test, y_test),
        shuffle=True
    )

    print("Treinando com Holdout (80% treino / 20% teste)")
    history = trainer.train(x_train, y_train)
    test_loss, test_acc = trainer.evaluate(x_test, y_test)
    print(f"Holdout - Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    return model, history, x_test, y_test, "Holdout"

def run_kfold_experiment(k=5):
    X, y = DataLoader.load_fashion_mnist("src/data/fashion_train.csv")

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_histories = []
    fold_accuracies = []
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Treinando Fold {fold_idx + 1}/{k}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

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
            shuffle=True
        )

        history = trainer.train(X_train, y_train)
        val_loss, val_acc = trainer.evaluate(X_val, y_val)

        fold_histories.append(history)
        fold_losses.append(val_loss)
        fold_accuracies.append(val_acc)

    # Faz média dos históricos
    avg_history = {
        "loss": np.mean([h["loss"] for h in fold_histories], axis=0).tolist(),
        "accuracy": np.mean([h["accuracy"] for h in fold_histories], axis=0).tolist(),
        "val_loss": np.mean([h["val_loss"] for h in fold_histories], axis=0).tolist(),
        "val_accuracy": np.mean([h["val_accuracy"] for h in fold_histories], axis=0).tolist(),
    }

    # Para KFold, testaremos no mesmo conjunto de teste do holdout
    x_test, y_test = DataLoader.load_fashion_mnist("src/data/fashion_test.csv")
    final_model = model  # último modelo treinado no último fold

    print(f"K-Fold ({k} folds) - Média Validation Accuracy: {np.mean(fold_accuracies):.4f} | Média Validation Loss: {np.mean(fold_losses):.4f}")

    return final_model, avg_history, x_test, y_test, f"{k}-Fold Cross Validation"

def plot_results(histories, titles):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for hist, title in zip(histories, titles):
        plt.plot(hist["loss"], label=f"Train Loss - {title}")
        plt.plot(hist["val_loss"], linestyle='--', label=f"Val Loss - {title}")
    plt.title("Loss por Época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    for hist, title in zip(histories, titles):
        plt.plot(hist["accuracy"], label=f"Train Acc - {title}")
        plt.plot(hist["val_accuracy"], linestyle='--', label=f"Val Acc - {title}")
    plt.title("Acurácia por Época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"imgs/compare_kfold_holdout_{timestamp}.png")
    plt.show()

def plot_confusion_matrix(model, x_test, y_test, title):
    y_pred = model.predict(x_test)
    cm = Metrics.confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
    plt.title(f"Matriz de Confusão - {title}")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Real")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"imgs/confusion_matrix_kfold_holdout_{title.replace(' ', '_').lower()}_{timestamp}.png")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = []
    
    model_holdout, hist_holdout, x_test, y_test, title_holdout = run_holdout_experiment()
    results.append((model_holdout, hist_holdout, x_test, y_test, title_holdout))

    model_kfold, hist_kfold, x_test, y_test, title_kfold = run_kfold_experiment(k=5)
    results.append((model_kfold, hist_kfold, x_test, y_test, title_kfold))

    plot_results([r[1] for r in results], [r[4] for r in results])

    for model, _, x_test, y_test, title in results:
        plot_confusion_matrix(model, x_test, y_test, title)
