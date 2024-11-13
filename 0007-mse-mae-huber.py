import torch

y_hat = torch.tensor([2.5, 0.0, 2.0, 8.0])
y = torch.tensor([3.0, -0.5, 2.0, 7.0])

# - - -  - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# CUSTOM MSE
def custom_MSE(y, y_hat):
    diff = y - y_hat
    squared = diff ** 2
    avg = squared.mean()
    return avg

# BUILT IN MSE
MSE = torch.nn.MSELoss()
loss = custom_MSE(y, y_hat)
print(loss)
loss = MSE(y, y_hat)
print(loss)

# - - -  - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# CUSTOM MAE
def custom_MAE(y, y_hat):
    diff = y - y_hat
    abs_diff = torch.abs(diff)
    avg = abs_diff.mean()
    return avg
# BUILT-IN MAE (L1 Loss in PyTorch)
MAE = torch.nn.L1Loss()
loss = custom_MAE(y, y_hat)
print(loss)
loss = MAE(y, y_hat)
print(loss)

# - - -  - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# CUSTOM HUBER LOSS
def custom_huber_loss(y, y_hat, delta=1.0):
    diff = y - y_hat
    abs_diff = torch.abs(diff)
    
    loss = torch.where(abs_diff < delta,
                       0.5 * (diff ** 2), #MSE
                       delta * (abs_diff - 0.5 * delta)) #MAE
    
    avg_loss = loss.mean()
    return avg_loss
  
# BUILT-IN HUBER LOSS
huber_loss_fn = torch.nn.SmoothL1Loss()
loss = custom_huber_loss(y, y_hat, delta=1.0)
print("Custom Huber Loss:", loss)
loss_builtin = huber_loss_fn(y_hat, y)
print("Built-in Huber Loss:", loss_builtin)
