# This is the original eleven line neural network code
# from the website/blog article.

# Looking at wikipedia, there is a lot of details hiding here that can be filled in. This is an example of a simple "backpropagation" network 

# Not exactly a copy/paste as I'm going to annotate the
# code with comments based on the notes on the webpage.

import numpy as np

# This represents the nonlinearity in the system. Could be any number
# of kinds of functions. This is a sigmoid which maps any value to a
# value between 0 and 1. It is used to convert numbers to
# probabilities and has desirable properties for training neural
# networks.

# So the properties are as follows
# - Real-valued and differentiable.
# - The sigmoid function here models the behavior of a neuron (firing or not, 1 or 0).
#
# Positive slope (in the derivative) shows that the neuron is firing.

# This particular sigmoid is also known as the logistic function.

# Hyperbolic tangent.

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))

# MAIN

# input dataset
X = np.array([ [0,0,1],
               [0,1,0],
               [0,1,1],
               [1,0,1],
               [1,1,0],
               [1,0,0] ])


# output dataset
y = np.array([[0,0,0,1,1,1]]).T

# Given this input/output, the goal is to recognize the
# strings that start with 1.

# What do these actually look like?
print(X)
print(y)

# seed random numbers to make calculation
# deterministic (just a good practice)

np.random.seed(1)

# The network is two layers connected by one layer
# of weights (the synapse).

# l0 -> synapse0 -> l1
# l0 is the first layer, specified by the input data.
# l1 is the hidden layer of the network.

# initialize weights randomly with mean 0
# This generates a column vector. The numpy description
# is that this returns random floats in the half-open interval
# of [0.0,1.0). Continuous uniform distribution.

# (3,1) says return a 3x1 array. The values here are then
# 2 * [0.0,1.0) - 1. Not sure how this defines the mean to be zero.

synapse0 = 2 * np.random.random((3,1)) - 1

print("synapse0")
print(synapse0)

# No xrange in python3.
#for iter in xrange(100000):
for iter in range(5):

    # forward propagation step
    layer0 = X

    # First multiply layer0 by the weights in synapse0 and then
    # feed that through the sigmoid function.

    # so y = nonlin(dot(x,w)) where dot in this case woudl be a matrix
    # multiplication

    layer1 = nonlin(np.dot(layer0,synapse0))

    print("layer1")
    print(layer1)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # error calculation
    layer1_error = y - layer1
    
    # multiply how much we missed by the
    # slope of the sigmoid at the values in layer1
    # positive slope represents activation.

    layer1_delta = layer1_error * nonlin(layer1,True)

    # update weights
    synapse0 += np.dot(layer0.T,layer1_delta)

# end loop
print("Output after training: ")
print(layer1)
print("Weights after training: ")
print(synapse0)
# Try out something it wasn't taught.
print("Trying [0,0,0].")
layer1 = nonlin(np.dot(np.array([0,0,0]),synapse0))
print(layer1)

print("Trying [0,1,1].")
layer1 = nonlin(np.dot(np.array([0,1,1]),synapse0))
print(layer1)

print("Trying [1,1,1].")
layer1 = nonlin(np.dot(np.array([1,1,1]),synapse0))
print(layer1)

print("Trying [1,0,0].")
layer1 = nonlin(np.dot(np.array([1,0,0]),synapse0))
print(layer1)

#END MAIN

