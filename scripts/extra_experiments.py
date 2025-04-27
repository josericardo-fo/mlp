import time, sys, os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_loader import DataLoader
from src.models.mlp import MLP
from src.trainers.trainer import Trainer
from src.utils.metrics import Metrics


def run_timed_training(batch_size, epochs=10):
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
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
    )

    start = time.time()
    history = trainer.train(x_train, y_train, verbose=False)
    duration = time.time() - start

    print(f"Batch size {batch_size} | Time: {duration:.2f}s")
    return history


def run_overfitting_test():
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
        epochs=100,
        validation_data=(x_test, y_test),
        shuffle=True,
    )

    print("Rodando experimento para avaliar overfitting com 100 épocas...")
    history = trainer.train(x_train, y_train)
    print("Salvando modelo treinado...")
    model.save("model.pkl")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Treino")
    plt.plot(history["val_loss"], label="Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.title("Loss por Época (100 épocas)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Treino")
    plt.plot(history["val_accuracy"], label="Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.title("Acurácia por Época (100 épocas)")
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"imgs/overfitting_analysis_{timestamp}.png")
    plt.show()


def visualizar_erros_classe_6():
    import matplotlib.pyplot as plt

    x_test, y_test = DataLoader.load_fashion_mnist("src/data/fashion_test.csv")
    model = MLP.load("model.pkl")  # Requer que um modelo treinado já tenha sido salvo
    y_pred = model.predict(x_test)

    y_true_class = np.argmax(y_test, axis=1)
    y_pred_class = np.argmax(y_pred, axis=1)

    erros_classe_6 = np.where((y_true_class == 6) & (y_pred_class != 6))[0]

    print(f"Total de erros na classe 6: {len(erros_classe_6)}")

    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(erros_classe_6[:10]):
        plt.subplot(2, 5, i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {y_pred_class[idx]}")
        plt.axis("off")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"imgs/erros_classe_6_{timestamp}.png")
    plt.show()


if __name__ == "__main__":
    print("\n⏱️ Comparando tempos de treino:")
    for bs in [1, 32, 512]:
        run_timed_training(bs)

    print("\n🔍 Rodando teste de overfitting (100 épocas):")
    run_overfitting_test()

    print("\n📸 Visualizando erros da classe 6 (modelo salvo necessário):")
    visualizar_erros_classe_6()