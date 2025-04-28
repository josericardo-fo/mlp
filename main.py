import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_loader import DataLoader
from src.models.mlp import MLP
from src.trainers.trainer import Trainer
from src.utils.metrics import Metrics


def main():
    # Verificar se os arquivos existem
    train_path = "data/fashion-mnist_train.csv"
    test_path = "data/fashion-mnist_test.csv"
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {test_path}")

    # Inicializar o DataLoader
    data_loader = DataLoader(batch_size=32, shuffle=True)

    # Carregar dados
    print("Carregando dados...")
    x_train, y_train = data_loader.load_fashion_mnist(train_path)
    x_test, y_test = data_loader.load_fashion_mnist(test_path)

    # Pré-processar dados
    print("Pré-processando dados...")
    x_train_processed, y_train_processed = data_loader.preprocess_data(x_train, y_train)
    x_test_processed, y_test_processed = data_loader.preprocess_data(x_test, y_test)

    # Achatar as imagens para o MLP (de 2D para 1D)
    x_train_processed = x_train_processed.reshape(x_train_processed.shape[0], -1)
    x_test_processed = x_test_processed.reshape(x_test_processed.shape[0], -1)

    # Criar mini-batches
    print("Criando mini-batches...")
    train_batches = data_loader.create_mini_batches(x_train_processed, y_train_processed)

    # Visualizar primeiro batch
    print("\nVisualizando primeiro batch de imagens...")
    data_loader.visualize_batch(
        train_batches[0][0].reshape(-1, 28, 28),
        train_batches[0][1],
        n_samples=5
    )

    # Fazer benchmark de pré-processamento
    print("\nFazendo benchmark de pré-processamento...")
    benchmark_results = data_loader.benchmark_preprocessing(x_train, y_train)
    
    # Mostrar algumas estatísticas
    print("\nEstatísticas dos dados:")
    print(f"Shape dos dados de treino: {x_train.shape}")
    print(f"Shape dos dados de teste: {x_test.shape}")
    print(f"Número de batches de treino: {len(train_batches)}")
    print(f"Tamanho de cada batch: {train_batches[0][0].shape}")

    print("\nIniciando treinamento do MLP em Fashion MNIST...")

    # Criar modelo
    print("Inicializando modelo...")
    model = MLP(
        input_size=784,  # Tamanho da camada de entrada (imgs MNIST 28x28)
        hidden_size=128,
        output_size=10,  # Número de classes (0-9)
        activation="relu",
        learning_rate=0.01,
        momentum=0.0,
        use_bias=True,
        weight_init="he",
    )

    # Configurar e treinar modelo
    print("Iniciando treinamento...")
    trainer = Trainer(
        model=model,
        batch_size=32,
        epochs=50,
        validation_data=(x_test_processed, y_test_processed),
    )

    history = trainer.train(x_train_processed, y_train_processed)

    # Avaliar modelo
    print("Avaliando modelo...")
    test_loss, test_acc = trainer.evaluate(x_test_processed, y_test_processed)
    print(f"Acurácia final no conjunto de teste: {test_acc:.2f}%")
    print(f"Loss final no conjunto de teste: {test_loss:.4f}")

    # Plotar resultados
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Treino")  # Modificado de 'train_loss' para 'loss'
    plt.plot(history["val_loss"], label="Validação")
    plt.title("Loss durante o treinamento")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Treino")
    plt.plot(history["val_accuracy"], label="Validação")
    plt.title("Acurácia durante o treinamento")
    plt.xlabel("Época")
    plt.ylabel("Acurácia (%)")
    plt.legend()

    # Criar pasta para imagens se não existir
    os.makedirs("imgs", exist_ok=True)

    # Salvar figura
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"imgs/results_{timestamp}.png")
    plt.show()

    # Gerar e mostrar matriz de confusão
    y_pred = model.predict(x_test_processed)
    cm = Metrics.confusion_matrix(y_test_processed, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Real")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )
    plt.savefig(f"imgs/confusion_matrix_{timestamp}.png")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
