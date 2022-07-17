import numpy as np

class NeuralNetwork:
    
    def __init__(self, *args):
        self.layerNumber = len(args) - 1
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.layerNumber):
            self.weights.append(np.random.randn(args[i+1], args[i]))
            self.biases.append(np.random.randn(args[i+1], 1))
            
            
    def feed_forward(self, x):
        for i in range(self.layerNumber):
            x = self.sigmoid(np.dot(self.weights[i], x) + self.biases[i])
            
        return x
    
    def train(self, x, y, learning_rate = 1):
        
        # Feed forward x and memorize the outputs and the activations
        outputs = []
        activations = [x]
        for i in range(self.layerNumber):
            outputs.append(np.dot(self.weights[i], activations[i]) + self.biases[i])
            activations.append(self.sigmoid(outputs[i]))
            
        # Find the errors
        errors = [y - activations[-1]]
        for i in range(self.layerNumber - 1):
            errors.append(np.dot(self.weights[self.layerNumber - 1 - i].T, errors[i]))
        errors.reverse()
        
        # Backpropagate the error
        weights_gradient = []
        biases_gradient = []
        
        for i in range(self.layerNumber):
            weights_gradient.append(np.dot(learning_rate * errors[i] * 
                                           self.d_sigmoid(outputs[i]), activations[i].T))
            
            biases_gradient.append(learning_rate * errors[i] * self.d_sigmoid(outputs[i]))
        
            self.weights[i] = self.weights[i] + weights_gradient[i]
            self.biases[i] = self.biases[i] + biases_gradient[i]
            
            
# =============================================================================
#     Activation Functions        
# =============================================================================
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
        
    def relu(self,x):
        return x if x > 0 else 0
    
    def d_relu(self,x):
        return 1 if x > 0 else 0
            
            