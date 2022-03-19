import numpy as np

#activation functions for output layer
def linear(z, derivate=False):
    a = z
    if derivate:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivate=False):
    a = 1/(1 + np.exp(-z))
    if derivate:
        da = np.ones(z.shape)
        return a, da
    return a

def softmax(z, derivate=False):
    e = np.exp(z - np.max(z, axis=0))
    a = e / np.sum(e, axis=0)
    if derivate:
        da = np.ones(z.shape)
        return a, da
    return a

# activation functions for hidden layers

def tanh(z, derivate = False):
    a = np.tanh(z)
    if derivate:
        da = (1 - a) * (1 + a)
        return a, da
    return a

def relu(z, derivate = False):
    a = z * (z >= 0)
    if derivate:
        da = np.array(z >= 0, dtype=float)
        return a, da
    return a

def logistic_hidden(z, derivate = False):
    a = 1 /(1 + np.exp(-z))
    if derivate:
        da = a * (1 - a)
        return a, da
    return a