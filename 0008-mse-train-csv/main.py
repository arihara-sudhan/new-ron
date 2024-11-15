"""
  CODE CONTRIBUTED BY: https://github.com/sahayaantonyA
"""
import csv , torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self,epochs,file_path,learning_rate=0.01):
        super(Network,self).__init__()
        self.x, self.y = self.csv_reader(file_path)
        self.hidden = nn.Linear(1,5)
        self.output = nn.Linear(5,1)
        self.relu = nn.ReLU()
        self.epochs = epochs
        self.lr = learning_rate

    def csv_reader(self,file_path):
        x = []
        y = []
        with open(file_path,'r') as file:
            data = csv.reader(file)
            next(data)
            for row in data:
                x.append([float(row[0])])
                y.append([float(row[1])])

        return torch.tensor(x),torch.tensor(y)
        
    def forward(self):
        x = self.relu(self.hidden(self.x))
        return self.output(x)

    def train(self):
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(),lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            y_hat = self.forward()
            error = loss_function(y_hat,self.y)
            error.backward()
            optimizer.step()
        
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs :<8} : Loss: {error.item():.4f}")

    def predict(self,x):
        with torch.no_grad():
            x_tensor = torch.tensor([[float(x)]])
            y = self.relu(self.hidden(x_tensor))
            print(f"Input: {x_tensor.item() :<5} : Output: {self.output(y).item() :.2f}")
                
model = Network(epochs=1000,file_path='./data_set.csv',learning_rate=0.02) 

print("Model")
print("===" * 8)
print(model,end='\n\n')
# print("----"*7)

print("Training")
print("===" * 8)
model.train()
print(end='\n\n')

print("Prediction")
print("===" * 8)
(model.predict(x=10))
