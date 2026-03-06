import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from block_graph import build_graph_payload
from validation_config import DEFAULT_RANDOM_SEED, DEFAULT_TRAIN_EPOCHS

try:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    DIGITS_DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    load_digits = None
    train_test_split = None
    StandardScaler = None
    DIGITS_DEPENDENCIES_AVAILABLE = False

class TinyDigitClassifier(nn.Module):
    """A simple two-layer MLP whose hidden activation matches the block-graph runtime."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _require_digits_dependencies():
    if not DIGITS_DEPENDENCIES_AVAILABLE:
        raise RuntimeError(
            "scikit-learn is required for the computational-storage digits validation. "
            "Install it via requirements.txt."
        )


def set_training_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_digits_split(test_size: float = 0.2, random_state: int = 42):
    _require_digits_dependencies()
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_digit_classifier(
    X_train,
    y_train,
    epochs: int = DEFAULT_TRAIN_EPOCHS,
    seed: int = DEFAULT_RANDOM_SEED,
    learning_rate: float = 0.01,
):
    set_training_seed(seed)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)

    model = TinyDigitClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

    return model


def evaluate_torch_model(model: nn.Module, X_test, y_test) -> float:
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    with torch.no_grad():
        test_out = model(X_test_t)
        preds = torch.argmax(test_out, dim=1)
        acc = (preds == y_test_t).float().mean().item()
    return float(acc)


def compile_model(model: TinyDigitClassifier, output_path: str):
    print("Extracting weights for Computational Storage serialization...")
    w1 = model.fc1.weight.detach().cpu().numpy().T
    w2 = model.fc2.weight.detach().cpu().numpy().T
    payload = build_graph_payload([w1, w2])

    with open(output_path, "wb") as f:
        f.write(payload)

    return {
        "output_path": output_path,
        "payload_size": len(payload),
        "layer_shapes": [w1.shape, w2.shape],
    }


def train_and_compile(
    output_path: str | None = None,
    epochs: int = DEFAULT_TRAIN_EPOCHS,
    seed: int = DEFAULT_RANDOM_SEED,
):
    print("Loading Digits dataset (8x8 images = 64 inputs)...")
    X_train, X_test, y_train, y_test = load_digits_split()

    print("Training model...")
    model = train_digit_classifier(X_train, y_train, epochs=epochs, seed=seed)
    torch_accuracy = evaluate_torch_model(model, X_test, y_test)
    print(f"Test Accuracy: {torch_accuracy * 100:.2f}%")

    if output_path is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(out_dir, "real_model.bin")

    compile_metrics = compile_model(model, output_path)
    print(f"\nCompiled REAL binary graph to {compile_metrics['output_path']}")
    print(f"Total size: {compile_metrics['payload_size']} bytes")
    print("Graph Structure:")
    print(f"  Block 0 (Offset 0x0) -> W1 {compile_metrics['layer_shapes'][0]} -> Points to Block 1")
    print(f"  Block 1 (Offset 0x80008) -> W2 {compile_metrics['layer_shapes'][1]} -> Points to 0x0 (End)")

    compile_metrics.update(
        {
            "torch_accuracy": torch_accuracy,
            "epochs": epochs,
            "seed": seed,
            "test_samples": len(y_test),
        }
    )
    return compile_metrics


if __name__ == "__main__":
    train_and_compile()
