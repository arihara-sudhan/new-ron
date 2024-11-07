import torch
import torch.nn as nn
import torch.optim as optim

class XORMLP(nn.Module):
    def __init__(self):
        super(XORMLP, self).__init__()
        self.hidden_one = nn.Linear(2, 8) 
        self.hidden_two = nn.Linear(8, 4) 
        self.output = nn.Linear(4, 1)      
        self.relu = nn.ReLU()              

    def forward(self, x):
        x = self.relu(self.hidden_one(x))
        x = self.relu(self.hidden_two(x))
        x = torch.sigmoid(self.output(x)) 
        return x

model = XORMLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

epochs = 2000
for epoch in range(epochs):
    output = model(X)
    loss = criterion(output, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    for i, sample in enumerate(X):
        output = model(sample)
        print(f'Input: {sample.tolist()}, Output: {output.item():.4f}, Expected: {Y[i].item()}')
