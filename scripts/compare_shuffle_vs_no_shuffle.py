import os, sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_loader import DataLoader
from src.models.mlp import MLP
from src.trainers.trainer import Trainer
from src.utils.metrics import Metrics

def run_experiment(shuffle, title):
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
        shuffle=shuffle,
    )

    print(f"Treinando com shuffle={shuffle} ({title})")
    history = trainer.train(x_train, y_train)
    test_loss, test_acc = trainer.evaluate(x_test, y_test)
    print(f"{title} - Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    return model, history, x_test, y_test, title

def plot_results(histories, titles):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for hist, title in zip(histories, titles):
        plt.plot(hist["loss"], label=f"Loss - {title}")
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
    plt.savefig(f"imgs/compare_shuffle_{timestamp}.png")
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
    plt.savefig(f"imgs/confusion_matrix_shuffle_{title.replace(' ', '_').lower()}_{timestamp}.png")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    configs = [(True, "Com Shuffle"), (False, "Sem Shuffle")]

    results = []
    for shuffle, title in configs:
        model, hist, x_test, y_test, label = run_experiment(shuffle, title)
        results.append((model, hist, x_test, y_test, label))

    plot_results([r[1] for r in results], [r[4] for r in results])

    for model, _, x_test, y_test, label in results:
        plot_confusion_matrix(model, x_test, y_test, label)