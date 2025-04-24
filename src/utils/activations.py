import numpy as np

class Activation:
    """
    Classe que implementa várias funções de ativação e suas derivadas.
    """
    
    @staticmethod
    def sigmoid(x):
        """Função de ativação sigmoid."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(x):
        """Derivada da função sigmoid."""
        sx = Activation.sigmoid(x)
        return sx * (1 - sx)

    @staticmethod
    def tanh(x):
        """Função de ativação tangente hiperbólica."""
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        """Derivada da função tangente hiperbólica."""
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        """Função de ativação ReLU (Rectified Linear Unit)."""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """Derivada da função ReLU."""
        return np.where(x > 0, 1, 0)
        
    @staticmethod
    def softmax(x):
        """Função de ativação softmax."""
        # Subtrai o máximo para evitar overflow numérico
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(x):
        """
        Derivada simplificada da função softmax.
        Observação: Esta implementação simplificada só é válida quando usada
        com a função de perda de entropia cruzada.
        """
        s = Activation.softmax(x)
        return s * (1 - s)
    
    @staticmethod
    def get_activation_and_derivative(activation_name):
        """
        Retorna a função de ativação e sua derivada com base no nome.
        
        Args:
            activation_name: Nome da função de ativação ('sigmoid', 'tanh', 'relu', 'softmax')
            
        Returns:
            Tupla contendo a função de ativação e sua derivada
        """
        activation_map = {
            'sigmoid': (Activation.sigmoid, Activation.sigmoid_derivative),
            'tanh': (Activation.tanh, Activation.tanh_derivative),
            'relu': (Activation.relu, Activation.relu_derivative),
            'softmax': (Activation.softmax, Activation.softmax_derivative)
        }
        
        if activation_name.lower() not in activation_map:
            raise ValueError(f"Função de ativação '{activation_name}' não implementada")
            
        return activation_map[activation_name.lower()]