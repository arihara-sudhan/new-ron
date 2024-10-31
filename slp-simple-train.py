w1, w2 = 0, 0
b = 0
inputs = [1, 0]
target = 0

def activation(z):
    return 1 if z >= 0 else 0

max_iterations = 10
for i in range(max_iterations):
    z = w1 * inputs[0] + w2 * inputs[1] + b
    output = activation(z)
    error = target - output
    print(f"Loss: {error} | w1: {w1} | w2: {w2} | b: {b}")
    if error == 0:
        break
    w1 += error * inputs[0]
    w2 += error * inputs[1]
    b += error

x1 = int(input("Enter Input Feature1: "))
x2 = int(input("Enter Input Feature2: "))
output = activation(w1*x1 + w2*x2 + b)
print(f"OUTPUT: {output}")
