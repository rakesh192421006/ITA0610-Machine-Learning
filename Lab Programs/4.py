import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

# Training data (XOR)
data = [
    ([0,0],[0]),
    ([0,1],[1]),
    ([1,0],[1]),
    ([1,1],[0])
]

# Initialize weights
w1 = random.random()
w2 = random.random()
w3 = random.random()
w4 = random.random()
b1 = random.random()
b2 = random.random()

lr = 0.5

# Training
for epoch in range(5000):
    for x,y in data:
        x1,x2 = x
        t = y[0]

        # ---- Forward pass ----
        h = sigmoid(x1*w1 + x2*w2 + b1)
        o = sigmoid(h*w3 + b2)

        # ---- Backpropagation ----
        error = t - o
        d_o = error * dsigmoid(o)
        d_h = d_o * w3 * dsigmoid(h)

        # ---- Update weights ----
        w3 += lr * d_o * h
        b2 += lr * d_o

        w1 += lr * d_h * x1
        w2 += lr * d_h * x2
        b1 += lr * d_h

# Testing
print("\nXOR Output")
for x,y in data:
    x1,x2 = x
    h = sigmoid(x1*w1 + x2*w2 + b1)
    o = sigmoid(h*w3 + b2)
    print(x,"->",round(o,3))
