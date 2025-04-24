import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.mlp import MLP
from src.utils.activations import Activation
from src.utils.losses import Losses
from src.data.data_loader import DataLoader
from src.trainers.trainer import Trainer
from src.utils.metrics import Metrics

def main():
    print("Iniciando treinamento do MLP em Fashion MNIST...")
    
    # Carregar dados
    print("Carregando dados...")
    x_train, y_train = DataLoader.load_fashion_mnist('src/data/fashion_train.csv')
    x_test, y_test = DataLoader.load_fashion_mnist('src/data/fashion_test.csv')
    
    # Obter funções de ativação
    relu_func, relu_deriv = Activation.get_activation_and_derivative('relu')
    softmax_func, softmax_deriv = Activation.get_activation_and_derivative('softmax')
    
    # Criar modelo
    print("Inicializando modelo...")
    model = MLP(
        input_size=784,           # Tamanho da camada de entrada (imgs MNIST 28x28)
        hidden_size=128,          # Tamanho da camada oculta
        output_size=10,           # Número de classes (0-9) 
        activation='relu',        # Função de ativação
        learning_rate=0.01,       # Taxa de aprendizado
        momentum=0.0,             # Momentum para otimização
        use_bias=True,            # Usar bias
        weight_init="xavier"      # Inicialização de pesos
    )
    
    # Configurar e treinar modelo
    print("Iniciando treinamento...")
    trainer = Trainer(
        model=model,
        batch_size=32,
        epochs=10,
        validation_data=(x_test, y_test)  # Passando os dados de validação aqui
    )

    history = trainer.train(x_train, y_train)
    
    # Avaliar modelo
    print("Avaliando modelo...")
    test_loss, test_acc = trainer.evaluate(x_test, y_test)
    print(f"Acurácia final no conjunto de teste: {test_acc:.2f}%")
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Treino')  # Modificado de 'train_loss' para 'loss'
    plt.plot(history['val_loss'], label='Validação')
    plt.title('Perda durante o treinamento')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Treino')  # Modificado de 'train_acc' para 'accuracy'
    plt.plot(history['val_accuracy'], label='Validação')  # Modificado de 'val_acc' para 'val_accuracy'
    plt.title('Acurácia durante o treinamento')
    plt.xlabel('Época')
    plt.ylabel('Acurácia (%)')
    plt.legend()
    
    # Salvar figura
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results_{timestamp}.png')
    plt.show()
    
    # Gerar e mostrar matriz de confusão
    y_pred = model.predict(x_test)
    cm = Metrics.confusion_matrix(y_test, y_pred)
    print("Matriz de confusão:")
    print(cm)

if __name__ == "__main__":
    main()