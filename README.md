# Redes Neurais Profundas: MLP (Multilayer Perceptron)

## Equipe

- Alunos: David O'Neil, Gabriel Orlow, José Ricardo, Murilo Álvares, Priscila Maia

## Descrição

Este projeto implementa uma rede neural do tipo Multilayer Perceptron (MLP) para classificação de imagens usando o dataset Fashion MNIST. A implementação é feita do zero usando apenas NumPy, sem frameworks de deep learning.

## Uso

Execute o script principal com:

```bash
python main.py
```

### Parâmetros disponíveis

```bash
python main.py --hidden_size 128 --activation relu --learning_rate 0.01 --batch_size 32 --epochs 50
```

Parâmetros:

- `--hidden_size`: Número de neurônios na camada oculta
- `--activation`: Função de ativação (relu, sigmoid, tanh)
- `--learning_rate`: Taxa de aprendizado
- `--momentum`: Valor do momentum para otimização
- `--batch_size`: Tamanho do batch para treinamento
- `--epochs`: Número de épocas de treinamento
- `--weight_init`: Método de inicialização de pesos (he, xavier, random)
- `--use_bias`: Usar bias nas camadas (flag)

## Métricas Implementadas

- Acurácia
- Perda (Cross-Entropy)
- Matriz de Confusão

## Visualizações

O treinamento gera automaticamente gráficos de:

- Curva de perda (treino vs. validação)
- Curva de acurácia (treino vs. validação)
- Matriz de confusão
