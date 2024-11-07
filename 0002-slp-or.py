x = [[1,1], [1,0], [0,1], [0,0]]
y = [1, 1, 1, 0]

def activate(wsum):
  return 0 if wsum < 0 else 1

def train(EPOCHS = 5):
  (w1, w2, BIAS) = (0, 0, 0)
  for epoch in range(EPOCHS):
    for i,data in enumerate(x):
      wsum = data[0]*w1 + data[1]*w2 + BIAS
      y_hat = activate(wsum)
      ERR = y[i] - y_hat
      print(f"Error: ${ERR}")
      if ERR != 0:
        w1 = w1 + ERR*data[0]
        w2 = w2 + ERR*data[1]
        BIAS = BIAS + ERR
  return w1, w2, BIAS

w1, w2, BIAS = train()
model = lambda x1, x2 : activate(w1*x1 + w2*x2 + BIAS)

print(model(0,0)) #0
print(model(0,1)) #1
print(model(1,0)) #1
print(model(1,0)) #1
