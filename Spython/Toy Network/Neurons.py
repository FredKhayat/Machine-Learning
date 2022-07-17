import random
import math


# =============================================================================
# Sigmoid Neuron
# =============================================================================
class Sigmoid():
    def __init__(self, neurons):
        self.neurons = []
        self.weights = []
        self.bias = random.gauss(0,1)
        self.value, self.delta = 0,0
        
        self.deltas_history = []
        self.weights_history = []
        
        for n in neurons:
            self.connect(n)
        
    
    def activate(self):
        self.value = 0
        for n,w in zip(self.neurons, self.weights):
            self.value += n.value * w
        self.value = self.sigmoid(self.value + self.bias)
        
        
    def backpropagate(self, learning_rate = 1):
        self.delta = self.delta * self.d_sigmoid(self.value)
        self.deltas_history.append(self.delta)
        self.weights_history.append(self.weights.copy())
            
        
        self.bias -= self.delta * learning_rate
        for idx, n in enumerate(self.neurons):
            n.delta += self.delta * self.weights[idx]
            self.weights[idx] -= self.delta * n.value * learning_rate
            
        self.delta = 0
    
    
    def connect(self, neuron):
        self.neurons.append(neuron)
        self.weights.append(random.gauss(0,1))
    
    def quadratic_cost(self, target):
        self.delta = self.value - target
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def d_sigmoid(self, x):
        return x * (1 - x)
    
        
# =============================================================================
# Watch Neuron
# =============================================================================
class Watch():
    def __init__(self, neurons, previous_watch = None):
        self.neurons = []
        self.previous_watch = previous_watch
        self.next_watch = None
        if(previous_watch): previous_watch.next_watch = self
            
        for n in neurons:
            self.neurons.append(n)
        
    def activate(self):
        for n in self.neurons:
            n.activate()
        if(self.next_watch):
            self.next_watch.activate()
            
    def backpropagate(self, learning_rate = 1):
        for n in self.neurons:
            n.backpropagate(learning_rate)
        if (self.previous_watch):
            self.previous_watch.backpropagate(learning_rate)
            

# =============================================================================
# Input Neuron
# =============================================================================
class Input():
    def __init__(self, v = 0):
        self.value = v
        self.delta = 0
    