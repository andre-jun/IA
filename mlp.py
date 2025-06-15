import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# ====== Hiperparâmetros ======
HIDDEN_SIZES = [64, 64]
LEARNING_RATE = 0.0005
MAX_EPOCHS = 100
LOSS_TARGET = 0.7
MODEL_OUTPUT = "choroMlp.pth"
NPZ_INPUT = "preProcessoTreinamento.npz"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregando os dados
data = np.load(NPZ_INPUT)
X = data["X"]
y = data["y"]

X = X.reshape(X.shape[0], -1)
input_size = X.shape[1]

# Convertendo para tensores do PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Criando Dataset e DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

modelo = ChoroMLP(input_size=input_size, hidden_sizes=HIDDEN_SIZES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

for epoch in range(1, MAX_EPOCHS + 1):
    modelo.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = modelo(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # avg_loss = running_loss / len(train_loader)

    print(f"Época {epoch}/{MAX_EPOCHS}, Loss: {running_loss:.4f}")
    if LOSS_TARGET is not None and running_loss <= LOSS_TARGET:
        print(f"🛑 Early stopping: loss {running_loss:.4f} ≤ target {LOSS_TARGET}")
        break


torch.save(modelo.state_dict(), "choroMlp.pth")
print(f"Modelo salvo como choroMlp.pth")
