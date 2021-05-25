"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np

from zknet.train import train
from zknet.nn import NeuralNet
from zknet.layers import Linear, Tanh,X

inputs = np.random.normal(scale=1,size=(1000,1))
w =np.array([[2]])
# inputs =np.array([[1],[2],[3]])
b =4
# 2x+1x+3
targets =  inputs @ w +b 
test = np.array([
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [99996]
])
net = NeuralNet([
    Linear(input_size=1, output_size=2),
#     Tanh(),

#     Linear(input_size=2, output_size=1)
])

train(net, inputs, targets)

for x in test:
    predicted = net.forward(x)
    print(x, predicted)
