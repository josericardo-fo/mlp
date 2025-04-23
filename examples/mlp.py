import argparse
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class MLP:
    """Implementação de uma Multilayer Perceptron (Rede Neural) usando NumPy."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: str = "sigmoid",
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        use_bias: bool = True,
        weight_init: str = "random",
        random_state: Optional[int] = None,
    ):
        """
        Inicializa a MLP.

        Args:
            input_size: Número de neurônios na camada de entrada
            hidden_size: Número de neurônios na camada oculta
            output_size: Número de neurônios na camada de saída
            activation: Função de ativação para camada oculta ('sigmoid', 'relu', 'tanh')
            learning_rate: Taxa de aprendizado para o backpropagation
            momentum: Termo de momentum para acelerar o treinamento
            use_bias: Se True, adiciona bias às camadas
            weight_init: Método de inicialização dos pesos ('random', 'xavier', 'he')
            random_state: Seed para reprodutibilidade
        """
        if random_state is not None:
            np.random.seed(random_state)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.use_bias = use_bias
        self.activation_name = activation

        # Configurar função de ativação
        self._setup_activation(activation)

        # Inicializar pesos
        self._initialize_weights(weight_init)

        # Inicializar termos de momentum
        self.prev_delta_weights_input_hidden = np.zeros((input_size, hidden_size))
        self.prev_delta_weights_hidden_output = np.zeros((hidden_size, output_size))
        self.prev_delta_bias_hidden = np.zeros((1, hidden_size))
        self.prev_delta_bias_output = np.zeros((1, output_size))

        # Métricas
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def _setup_activation(self, activation: str) -> None:
        """Configura a função de ativação e sua derivada."""
        if activation == "sigmoid":
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == "relu":
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation == "tanh":
            self.activation = self.tanh
            self.activation_derivative = self.tanh_derivative
        else:
            raise ValueError(f"Função de ativação {activation} não suportada.")

    def _initialize_weights(self, method: str) -> None:
        """Inicializa os pesos de acordo com o método escolhido."""
        if method == "random":
            self.weights_input_hidden = (
                np.random.randn(self.input_size, self.hidden_size) * 0.01
            )
            self.weights_hidden_output = (
                np.random.randn(self.hidden_size, self.output_size) * 0.01
            )
        elif method == "xavier":  # Bom para sigmoid/tanh

            def xavier_init(n_in, n_out):
                return np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)

            self.weights_input_hidden = xavier_init(self.input_size, self.hidden_size)
            self.weights_hidden_output = xavier_init(self.hidden_size, self.output_size)
        elif method == "he":  # Bom para ReLU

            def he_init(n_in, n_out):
                return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

            self.weights_input_hidden = he_init(self.input_size, self.hidden_size)
            self.weights_hidden_output = he_init(self.hidden_size, self.output_size)
        else:
            raise ValueError(f"Método de inicialização {method} não suportado.")

        if self.use_bias:
            self.bias_hidden = np.zeros((1, self.hidden_size))
            self.bias_output = np.zeros((1, self.output_size))

    # Funções de ativação
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Função de ativação sigmoid."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clipping para evitar overflow

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada da função sigmoid."""
        sx = self.sigmoid(x)
        return sx * (1 - sx)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Função de ativação tangente hiperbólica."""
        return np.tanh(x)

    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada da função tanh."""
        return 1 - np.tanh(x) ** 2

    def relu(self, x: np.ndarray) -> np.ndarray:
        """Função de ativação ReLU."""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada da função ReLU."""
        return np.where(x > 0, 1, 0)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Função softmax para saída da rede."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza o passo forward pela rede.

        Args:
            X: Matriz de características de entrada (amostras x features)

        Returns:
            A saída da camada final após softmax
        """
        # Camada oculta
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        if self.use_bias:
            self.hidden_input += self.bias_hidden
        self.hidden_output = self.activation(self.hidden_input)

        # Camada de saída
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        if self.use_bias:
            self.final_input += self.bias_output
        self.final_output = self.softmax(self.final_input)

        return self.final_output

    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """
        Realiza o passo backward e atualiza os pesos.

        Args:
            X: Matriz de características de entrada
            y: Matriz de rótulos one-hot encoded
            output: Saída predita pelo modelo
        """
        # Erro na camada de saída
        output_error = output - y

        # Erro na camada oculta
        if self.activation_name == "sigmoid":
            hidden_error = (
                np.dot(output_error, self.weights_hidden_output.T)
                * self.hidden_output
                * (1 - self.hidden_output)
            )
        else:
            hidden_error = np.dot(
                output_error, self.weights_hidden_output.T
            ) * self.activation_derivative(self.hidden_input)

        # Calcular gradientes
        delta_weights_hidden_output = np.dot(self.hidden_output.T, output_error)
        delta_weights_input_hidden = np.dot(X.T, hidden_error)

        # Atualizar pesos com momentum
        self.weights_hidden_output -= (
            self.learning_rate * delta_weights_hidden_output
            + self.momentum * self.prev_delta_weights_hidden_output
        )
        self.weights_input_hidden -= (
            self.learning_rate * delta_weights_input_hidden
            + self.momentum * self.prev_delta_weights_input_hidden
        )

        # Salvar para momentum
        self.prev_delta_weights_hidden_output = delta_weights_hidden_output
        self.prev_delta_weights_input_hidden = delta_weights_input_hidden

        # Atualizar bias se necessário
        if self.use_bias:
            delta_bias_output = np.sum(output_error, axis=0, keepdims=True)
            delta_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

            self.bias_output -= (
                self.learning_rate * delta_bias_output
                + self.momentum * self.prev_delta_bias_output
            )
            self.bias_hidden -= (
                self.learning_rate * delta_bias_hidden
                + self.momentum * self.prev_delta_bias_hidden
            )

            self.prev_delta_bias_output = delta_bias_output
            self.prev_delta_bias_hidden = delta_bias_hidden

    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula a perda de entropia cruzada."""
        # Adicionando pequeno valor para evitar log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula acurácia para predições."""
        y_true_class = np.argmax(y_true, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        return np.mean(y_true_class == y_pred_class)

    def train_batch(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Treina o modelo com um batch de dados.

        Args:
            X: Matriz de características
            y: Matriz de rótulos (one-hot encoded)

        Returns:
            Tuple contendo (loss, accuracy)
        """
        # Forward pass
        output = self.forward(X)

        # Backward pass
        self.backward(X, y, output)

        # Calcular métricas
        loss = self.calculate_loss(y, output)
        accuracy = self.calculate_accuracy(y, output)

        return loss, accuracy

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
        log_interval: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Treina o modelo por um número específico de épocas.

        Args:
            X: Matriz de características de treinamento
            y: Matriz de rótulos de treinamento (one-hot encoded)
            epochs: Número de épocas para treinar
            batch_size: Tamanho do mini-batch
            validation_data: Tupla opcional (X_val, y_val) para validação
            verbose: Se True, exibe progresso do treinamento
            log_interval: Intervalo para exibir log

        Returns:
            Dicionário contendo o histórico de treinamento
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        start_time = time.time()

        for epoch in range(epochs):
            # Embaralhar dados a cada época
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            epoch_acc = 0

            # Processar mini-batches
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                batch_loss, batch_acc = self.train_batch(X_batch, y_batch)
                epoch_loss += batch_loss * (end_idx - start_idx)
                epoch_acc += batch_acc * (end_idx - start_idx)

            # Calcular média
            epoch_loss /= n_samples
            epoch_acc /= n_samples

            # Validação se fornecida
            val_loss, val_acc = None, None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_predictions = self.forward(X_val)
                val_loss = self.calculate_loss(y_val, val_predictions)
                val_acc = self.calculate_accuracy(y_val, val_predictions)

            # Atualizar histórico
            self.training_history["loss"].append(epoch_loss)
            self.training_history["accuracy"].append(epoch_acc)
            if validation_data is not None:
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_accuracy"].append(val_acc)

            # Mostrar progresso
            if verbose and (epoch + 1) % log_interval == 0:
                elapsed_time = time.time() - start_time
                val_str = (
                    f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    if validation_data is not None
                    else ""
                )
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}{val_str}, Time: {elapsed_time:.2f}s"
                )

        if verbose:
            print(f"Treinamento finalizado em {time.time() - start_time:.2f} segundos")

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predições para os dados de entrada.

        Args:
            X: Matriz de características

        Returns:
            Matriz de probabilidades
        """
        return self.forward(X)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz as classes para os dados de entrada.

        Args:
            X: Matriz de características

        Returns:
            Array de classes preditas
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Avalia o modelo nos dados fornecidos.

        Args:
            X: Matriz de características
            y: Matriz de rótulos (one-hot encoded)

        Returns:
            Tuple contendo (loss, accuracy)
        """
        predictions = self.forward(X)
        loss = self.calculate_loss(y, predictions)
        accuracy = self.calculate_accuracy(y, predictions)
        return loss, accuracy

    def save(self, filepath: str) -> None:
        """
        Salva o modelo em um arquivo.

        Args:
            filepath: Caminho do arquivo
        """
        model_data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "weights_input_hidden": self.weights_input_hidden,
            "weights_hidden_output": self.weights_hidden_output,
            "bias_hidden": self.bias_hidden if self.use_bias else None,
            "bias_output": self.bias_output if self.use_bias else None,
            "activation": self.activation_name,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "use_bias": self.use_bias,
            "training_history": self.training_history,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str) -> "MLP":
        """
        Carrega um modelo de um arquivo.

        Args:
            filepath: Caminho do arquivo

        Returns:
            Instância carregada de MLP
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Criar instância
        model = cls(
            input_size=model_data["input_size"],
            hidden_size=model_data["hidden_size"],
            output_size=model_data["output_size"],
            activation=model_data["activation"],
            learning_rate=model_data["learning_rate"],
            momentum=model_data["momentum"],
            use_bias=model_data["use_bias"],
        )

        # Restaurar pesos
        model.weights_input_hidden = model_data["weights_input_hidden"]
        model.weights_hidden_output = model_data["weights_hidden_output"]

        if model.use_bias:
            model.bias_hidden = model_data["bias_hidden"]
            model.bias_output = model_data["bias_output"]

        # Restaurar histórico
        if "training_history" in model_data:
            model.training_history = model_data["training_history"]

        return model


def load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega o dataset Fashion MNIST.

    Returns:
        Tupla contendo (X_train, y_train, X_test, y_test)
    """
    train_df = pd.read_csv("data/fashion_train.csv")
    test_df = pd.read_csv("data/fashion_test.csv")

    # Separar features e labels
    y_train = train_df["label"].values
    X_train = train_df.drop("label", axis=1).values / 255.0  # Normalizar os dados

    y_test = test_df["label"].values
    X_test = test_df.drop("label", axis=1).values / 255.0  # Normalizar os dados

    # Converter para one-hot encoding
    y_train_onehot = np.zeros((y_train.size, 10))
    y_train_onehot[np.arange(y_train.size), y_train] = 1

    y_test_onehot = np.zeros((y_test.size, 10))
    y_test_onehot[np.arange(y_test.size), y_test] = 1

    return X_train, y_train_onehot, X_test, y_test_onehot


def main():
    """Função principal para executar via linha de comando."""
    parser = argparse.ArgumentParser(
        description="Treinamento e avaliação de MLP para Fashion MNIST"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Comando de treinamento
    train_parser = subparsers.add_parser("train", help="Treinar um modelo MLP")
    train_parser.add_argument(
        "--input-size", type=int, default=784, help="Tamanho da camada de entrada"
    )
    train_parser.add_argument(
        "--hidden-size", type=int, default=128, help="Tamanho da camada oculta"
    )
    train_parser.add_argument(
        "--output-size", type=int, default=10, help="Tamanho da camada de saída"
    )
    train_parser.add_argument(
        "--activation",
        type=str,
        choices=["sigmoid", "relu", "tanh"],
        default="sigmoid",
        help="Função de ativação para camada oculta",
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Taxa de aprendizado"
    )
    train_parser.add_argument(
        "--momentum", type=float, default=0.0, help="Valor de momentum"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Número de épocas para treinamento"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Tamanho do mini-batch"
    )
    train_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Proporção de dados para validação (0.0-1.0)",
    )
    train_parser.add_argument(
        "--weight-init",
        type=str,
        choices=["random", "xavier", "he"],
        default="random",
        help="Método de inicialização dos pesos",
    )
    train_parser.add_argument(
        "--no-bias", action="store_false", dest="use_bias", help="Não utilizar bias"
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default="model.pkl",
        help="Caminho para salvar o modelo",
    )
    train_parser.add_argument("--seed", type=int, help="Seed para reprodutibilidade")

    # Comando de avaliação
    eval_parser = subparsers.add_parser(
        "evaluate", help="Avaliar um modelo MLP treinado"
    )
    eval_parser.add_argument(
        "--model-path", type=str, required=True, help="Caminho para o modelo"
    )

    # Comando de predição
    predict_parser = subparsers.add_parser(
        "predict", help="Realizar predições com um modelo MLP"
    )
    predict_parser.add_argument(
        "--model-path", type=str, required=True, help="Caminho para o modelo"
    )
    predict_parser.add_argument(
        "--num-samples", type=int, default=10, help="Número de amostras para predizer"
    )

    args = parser.parse_args()

    if args.command == "train":
        print("Carregando dados...")
        X_train, y_train, X_test, y_test = load_fashion_mnist()

        # Split de validação
        if args.validation_split > 0:
            val_size = int(X_train.shape[0] * args.validation_split)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        print(
            f"Criando MLP com {args.input_size} entradas, {args.hidden_size} neurônios ocultos, {args.output_size} saídas"
        )
        print(
            f"Ativação: {args.activation}, Taxa de aprendizado: {args.learning_rate}, Momentum: {args.momentum}"
        )

        mlp = MLP(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            activation=args.activation,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            use_bias=args.use_bias,
            weight_init=args.weight_init,
            random_state=args.seed,
        )

        print(
            f"Iniciando treinamento por {args.epochs} épocas com batch size {args.batch_size}..."
        )
        mlp.train(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=validation_data,
            verbose=True,
        )

        # Avaliar no conjunto de teste
        test_loss, test_acc = mlp.evaluate(X_test, y_test)
        print(
            f"Avaliação no conjunto de teste - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}"
        )

        # Salvar modelo
        mlp.save(args.model_path)
        print(f"Modelo salvo em {args.model_path}")

    elif args.command == "evaluate":
        print(f"Carregando modelo de {args.model_path}...")
        mlp = MLP.load(args.model_path)

        print("Carregando dados de teste...")
        _, _, X_test, y_test = load_fashion_mnist()

        # Avaliar no conjunto de teste
        test_loss, test_acc = mlp.evaluate(X_test, y_test)
        print(
            f"Avaliação no conjunto de teste - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}"
        )

    elif args.command == "predict":
        print(f"Carregando modelo de {args.model_path}...")
        mlp = MLP.load(args.model_path)

        print("Carregando dados de teste...")
        _, _, X_test, y_test = load_fashion_mnist()

        # Selecionar algumas amostras aleatórias
        indices = np.random.choice(len(X_test), size=args.num_samples, replace=False)
        X_samples = X_test[indices]
        y_true = np.argmax(y_test[indices], axis=1)

        # Fazer predições
        y_pred = mlp.predict_classes(X_samples)

        # Mapear rótulos para nomes de classes
        class_names = [
            "Camiseta",
            "Calça",
            "Pulôver",
            "Vestido",
            "Casaco",
            "Sandália",
            "Camisa",
            "Tênis",
            "Bolsa",
            "Bota",
        ]

        # Mostrar resultados
        print("\nResultados de predição:")
        print(f"{'Índice':<10}{'Classe real':<20}{'Predição':<20}{'Correto'}")
        print("-" * 60)

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            correct = "✓" if true == pred else "✗"
            print(
                f"{indices[i]:<10}{class_names[true]:<20}{class_names[pred]:<20}{correct}"
            )

        accuracy = np.mean(y_true == y_pred)
        print(f"\nAcurácia nas amostras: {accuracy:.2f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
