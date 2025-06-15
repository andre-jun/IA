import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader

HIDDEN_SIZES = [64, 64]

class ChoroMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 2)
        )

    def forward(self, x):
        return self.net(x)

def evaluate(model_path, npz_path):
    print(f"Loading data from {npz_path}")
    data = np.load(npz_path)
    X = data['X']
    y = data['y']

    X = X.reshape(X.shape[0], -1)
    input_size = X.shape[1]

    print(f"Loading model from {model_path}")
    model = ChoroMLP(input_size=input_size, hidden_sizes=HIDDEN_SIZES)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=1)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (x, y_true) in enumerate(loader):
            out = model(x)
            pred = torch.argmax(out, dim=1).item()
            label = y_true.item()
            result = "✓ CORRECT" if pred == label else "✗ WRONG"
            pred_str = "yes" if pred == 1 else "no"
            label_str = "yes" if label == 1 else "no"
            print(f"[{result}] Sample {i} | Truth: {label_str} | Predicted: {pred_str}")
            correct += (pred == label)
            total += 1

    acc = 100 * correct / total
    print(f"\nOverall Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_cnn.py choroCnn.pth preProcesso.npz")
        sys.exit(1)

    model_file = sys.argv[1]
    npz_file = sys.argv[2]
    evaluate(model_file, npz_file)

