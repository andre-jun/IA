import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class ChoroCNN(nn.Module):
    def __init__(self):
        super(ChoroCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 64, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [16, 64, 129]
        x = self.pool(F.relu(self.conv2(x)))  # -> [32, 32, 64]
        x = x.view(x.size(0), -1)             # -> [batch, 32*32*64]
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Carregando os dados
data = np.load("preProcesso.npz")
X = data["X"]
y = data["y"]

# Adicionar canal (CNNs esperam 1xHxW)
X = X[:, np.newaxis, :, :]  # [N, 1, H, W]

# Convertendo para tensores do PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Criando Dataset e DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Abaixo disso 'e oq temq mudar pra acertar o modelo pq agr ta provavelmente dando overfitting(n testei)
#[bookmark]
modelo = ChoroCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)

n_epochs = 10
for epoch in range(n_epochs):
    modelo.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = modelo(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Época {epoch+1}/{n_epochs}, Loss: {running_loss:.4f}")

torch.save(modelo.state_dict(), "choroCnn.pth")
print(f"Modelo salvo como choroCnn.pth")
