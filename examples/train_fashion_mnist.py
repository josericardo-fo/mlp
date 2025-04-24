from src.data.data_loader import load_fashion_mnist
from src.models.mlp import MLP
from src.trainers.trainer import Trainer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train MLP on Fashion MNIST")
    parser.add_argument("--input-size", type=int, default=784, help="Input layer size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--output-size", type=int, default=10, help="Output layer size")
    parser.add_argument("--activation", type=str, choices=["sigmoid", "relu", "tanh"], default="sigmoid", help="Activation function")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum value")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--weight-init", type=str, choices=["random", "xavier", "he"], default="random", help="Weight initialization method")
    parser.add_argument("--no-bias", action="store_false", dest="use_bias", help="Disable bias")
    parser.add_argument("--model-path", type=str, default="model.pkl", help="Path to save the model")
    parser.add_argument("--seed", type=int, help="Seed for reproducibility")

    args = parser.parse_args()

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()

    if args.validation_split > 0:
        val_size = int(X_train.shape[0] * args.validation_split)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        validation_data = (X_val, y_val)
    else:
        validation_data = None

    print(f"Creating MLP with {args.input_size} inputs, {args.hidden_size} hidden neurons, {args.output_size} outputs")
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

    trainer = Trainer(model=mlp)

    print(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}...")
    trainer.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=validation_data)

    test_loss, test_acc = trainer.evaluate(X_test, y_test)
    print(f"Test evaluation - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    mlp.save(args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main()