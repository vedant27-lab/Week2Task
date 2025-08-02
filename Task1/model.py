import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #Defining weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1/(1+np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues*(self.output*(1-self.output))
    
class Loss_MSE:  #Mean Squred Error
    def forward(self, y_pred, y_true)->int: 
        self.samples = len(y_pred)
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true - y_pred)**2)
    def backward(self, y_pred, y_true):
        self.dinputs = -2*(y_true-y_pred)/self.samples

class SGD: #Stocastic Gradient Descent
    def __init__(self, learning_rate=0.01) -> None:
        self.learning_rate = learning_rate
    
    def update(self, layer):
        layer.weights -= self.learning_rate*layer.dweights
        layer.biases -= self.learning_rate*layer.dbiases