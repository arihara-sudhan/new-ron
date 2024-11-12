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

def train_with_sgd_momentum(model, X, y, epochs=1000, lr=0.1, momentum=0.9):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, losses

def train_with_nesterov(model, X, y, epochs=1000, lr=0.1, momentum=0.9):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    losses = []
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, losses

def train_with_adagrad(model, X, y, epochs=1000, lr=0.1):
    criterion = nn.BCELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, losses

def train_with_rmsprop(model, X, y, epochs=1000, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, losses

def train_with_adam(model, X, y, epochs=1000, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
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
trained_model, losses_sgd_momentum = train_with_sgd_momentum(model, X, y)
accuracy = evaluate(trained_model, X, y)
print(f"Accuracy with SGD with Momentum: {accuracy * 100:.2f}%")

model = XORModel()
trained_model, losses_nesterov = train_with_nesterov(model, X, y)
accuracy = evaluate(trained_model, X, y)
print(f"Accuracy with Nesterov: {accuracy * 100:.2f}%")

model = XORModel()
trained_model, losses_adagrad = train_with_adagrad(model, X, y)
accuracy = evaluate(trained_model, X, y)
print(f"Accuracy with Adagrad: {accuracy * 100:.2f}%")

model = XORModel()
trained_model, losses_rmsprop = train_with_rmsprop(model, X, y)
accuracy = evaluate(trained_model, X, y)
print(f"Accuracy with RMSprop: {accuracy * 100:.2f}%")

model = XORModel()
trained_model, losses_adam = train_with_adam(model, X, y)
accuracy = evaluate(trained_model, X, y)
print(f"Accuracy with Adam: {accuracy * 100:.2f}%")

plt.figure(figsize=(14, 7))
plt.plot(losses_sgd_momentum, label='SGD with Momentum')
plt.plot(losses_nesterov, label='Nesterov')
plt.plot(losses_adagrad, label='Adagrad')
plt.plot(losses_rmsprop, label='RMSprop')
plt.plot(losses_adam, label='Adam')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves for Different Optimizers')
plt.legend()
plt.show()
