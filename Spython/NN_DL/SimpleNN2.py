import numpy as np

class SimpleNN2:
    
    def __init__(self, inputNumber, hidenNumber, outputNumber):
        self.inputNumber = inputNumber
        self.hidenNumber = hidenNumber
        self.outputNumber = outputNumber
        
        self.weight_1 = np.random.randn(hidenNumber, inputNumber)
        self.weight_2 = np.random.randn(outputNumber, hidenNumber)
        self.bias_1 = np.random.randn(hidenNumber, 1)
        self.bias_2 = np.random.randn(outputNumber, 1)
        
    def feed_forward(self, inputs):
        a_1 = self.sigmoid(np.dot(self.weight_1, inputs) + self.bias_1)
        a_2 = self.sigmoid(np.dot(self.weight_2, a_1) + self.bias_2)
        return a_2
    
    def train(self, inputs, targets, learning_rate = 1):
        a_1 = self.sigmoid(np.dot(self.weight_1, inputs) + self.bias_1)
        a_2 = self.sigmoid(np.dot(self.weight_2, a_1) + self.bias_2)
        
        E = a_2 - targets
        
        delta_b_2 = E * self.d_sigmoid(a_2)
        delta_w_2 = np.dot(delta_b_2, a_1.T)
        
        delta_b_1 = np.dot(self.weight_2.T, delta_b_2) * self.d_sigmoid(a_1)
        delta_w_1 = np.dot(delta_b_1, inputs.T)
        
        self.weight_2 = self.weight_2 - delta_w_2 * learning_rate
        self.bias_2 = self.bias_2 - delta_b_2 * learning_rate
        self.weight_1 = self.weight_1 - delta_w_1 * learning_rate
        self.bias_1 = self.bias_1 - delta_b_1 * learning_rate
        
    
    def sigmoid(self, x):
        return 1/(1+ np.exp(-x))
    
    def d_sigmoid(self, x):
        return x * (1 - x)
        