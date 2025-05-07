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

def normalize_data(X_train, X_test, method='z-score'):
    """
    Normalize input data using the specified method.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training data to be normalized
    X_test : np.ndarray
        Test data to be normalized
    method : str
        Normalization method: 'min-max' or 'z-score'
        
    Returns:
    --------
    tuple: (normalized X_train, normalized X_test)
    """
    if method == 'min-max':
        X_train_norm = X_train / 255.0
        X_test_norm = X_test / 255.0
    elif method == 'z-score':
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std = np.where(std == 0, 1, std)  
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
    else:
        raise ValueError("Método de normalização inválido. Use 'min-max' ou 'z-score'.")
    return X_train_norm, X_test_norm        

def main():
    print("Iniciando treinamento do MLP em Fashion MNIST...")

    # Carregar dados
    print("Carregando dados...")
    x_train, y_train = DataLoader.load_fashion_mnist("src/data/fashion_train.csv")
    x_test, y_test = DataLoader.load_fashion_mnist("src/data/fashion_test.csv")

    # Normalizar dados
    print("Normalizando dados de entrada...")
    normalization_method = 'z-score'  # ou 'z-score'
    x_train, x_test = normalize_data(x_train, x_test, method=normalization_method)

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
        optimizer = "adam",
        l1_lambda=0.0001,
        l2_lambda=0.001,
        dropout_rate=0.2,
    )

    # Configurar e treinar modelo
    print("Iniciando treinamento...")
    trainer = Trainer(
        model=model,
        batch_size=32,
        epochs=50,
        validation_data=(x_test, y_test),  # Passando os dados de validação aqui
    )

    history = trainer.train(x_train, y_train)

    # Avaliar modelo
    print("Avaliando modelo...")
    test_loss, test_acc = trainer.evaluate(x_test, y_test)
    print(f"Acurácia final no conjunto de teste: {test_acc:.2f}%")
    print(f"Loss final no conjunto de teste: {test_loss:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Treino")  # Modificado de 'train_loss' para 'loss'
    plt.plot(history["val_loss"], label="Validação")
    plt.title("Loss durante o treinamento")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        history["accuracy"], label="Treino"
    )  # Modificado de 'train_acc' para 'accuracy'
    plt.plot(
        history["val_accuracy"], label="Validação"
    )  # Modificado de 'val_acc' para 'val_accuracy'
    plt.title("Acurácia durante o treinamento")
    plt.xlabel("Época")
    plt.ylabel("Acurácia (%)")
    plt.legend()

    # Salvar figura
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"imgs/results_{timestamp}.png")
    plt.show()

    # Gerar e mostrar matriz de confusão
    y_pred = model.predict(x_test)
    cm = Metrics.confusion_matrix(y_test, y_pred)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"imgs/confusion_matrix_{timestamp}.png")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
