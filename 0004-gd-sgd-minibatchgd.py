import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

def train_with_batch_gd(model, optimizer, X, y, epochs=500, lr=0.1):
    criterion = nn.BCELoss()
    optimizer = optimizer(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, losses

def train_with_sgd(model, optimizer, X, y, epochs=500, lr=0.1):
    criterion = nn.BCELoss()
    optimizer = optimizer(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        for i in range(len(X)):
            x_sample = X[i:i+1]
            y_sample = y[i:i+1]
            outputs = model(x_sample)
            loss = criterion(outputs, y_sample)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, losses

def train_with_mini_batch_gd(model, optimizer, X, y, batch_size=2, epochs=500, lr=0.1):
    criterion = nn.BCELoss()
    optimizer = optimizer(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, losses

def evaluate(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == y).float().mean()
    return accuracy.item()

model = XORModel()
optimizer = optim.SGD

trained_model, losses_batch_gd = train_with_batch_gd(model, optimizer, X, y, epochs=1000, lr=0.1)
accuracy = evaluate(trained_model, X, y)
print(f"Accuracy with Batch Gradient Descent: {accuracy * 100:.2f}%")

model = XORModel()
trained_model, losses_sgd = train_with_sgd(model, optimizer, X, y, epochs=1000, lr=0.1)
accuracy = evaluate(trained_model, X, y)
print(f"Accuracy with Stochastic Gradient Descent: {accuracy * 100:.2f}%")

model = XORModel()
trained_model, losses_mini_batch_gd = train_with_mini_batch_gd(model, optimizer, X, y, batch_size=2, epochs=1000, lr=0.1)
accuracy = evaluate(trained_model, X, y)
print(f"Accuracy with Mini-Batch Gradient Descent: {accuracy * 100:.2f}%")

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(losses_batch_gd, label='Batch Gradient Descent')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Batch Gradient Descent')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(losses_sgd, label='Stochastic Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Stochastic Gradient Descent')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(losses_mini_batch_gd, label='Mini-Batch Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Mini-Batch Gradient Descent')
plt.legend()

plt.tight_layout()
plt.show()
