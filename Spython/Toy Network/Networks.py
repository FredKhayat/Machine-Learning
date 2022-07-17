import Neurons

class SigmoidNetwork():
    def __init__(self, layers):
        self.neurons = [[] for i in range(len(layers))]
        self.watches = [None]
        
        # Add input neurons
        self.neurons[0] = [Neurons.Input() for i in range(layers[0])]
            
        for idx, l in enumerate(layers[1:]):
            # Add sigmoid neurons
            self.neurons[idx+1] = [Neurons.Sigmoid(self.neurons[idx]) for i in range(l)]
            # Add watch neurons
            self.watches.append(Neurons.Watch(self.neurons[idx+1], self.watches[idx]))
        
        # CLean up
        self.watches.remove(None)          
        
        
    def feedforward(self, inputs):
        # Set values to input neurons
        for idx, x in enumerate(inputs):
            self.neurons[0][idx].value = x
            
        self.watches[0].activate()
        output = [n.value for n in self.neurons[-1]]
        return output
    
    
    def backpropagate(self, x, y, learning_rate = 1):
        # Feedforward
        for idx, v in enumerate(x):
            self.neurons[0][idx].value = v
        self.watches[0].activate()
        
        for idx, n in enumerate(self.neurons[-1]):
            n.quadratic_cost(y[idx])
        
        self.watches[-1].backpropagate(learning_rate)
        
        
            