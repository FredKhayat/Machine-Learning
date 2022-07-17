from InfoWindow import InfoWindow
import pygame
import pygame_gui as pygui
import Neurons2
from GUI.Utils import ORANGE, GREEN, BLUE, RED, WHITE, BIAS_FONT, WEIGHT_FONT, smooth_increase, WEIGHT_COL

class Neurons():
    SIGMOID = 'Sigmoid'
    INPUT = 'Input'
    OUTPUT = 'Output'
    
    RACES = {SIGMOID : [Neurons2.Sigmoid, ORANGE],
             INPUT : [Neurons2.Input, GREEN],
             OUTPUT : [Neurons2.Sigmoid, BLUE]}
    
    RESOLUTION = 40
    WEIGHT_RATIO = 0.35
    neurons = []
    input_neurons = []
    output_neurons = []

    def __init__(self, pos, radius, race): 
        self.color = Neurons.RACES[race][1]
        self.neuron = Neurons.RACES[race][0]()
        self.race = race
        self.values_plot = [[0 for i in range(Neurons.RESOLUTION)] for j in range(Neurons.RESOLUTION)]
        
        self.pos = pos
        self.radius = radius
        self.rect = pygame.Rect(pos[0] - 2 * radius, pos[1] - 2 * radius, radius * 4, radius * 4)
        
        self.previous_neurons = []
    
    
    def connect(self, other):
        if other not in self.previous_neurons and other != self:
            self.previous_neurons.append(other)
            self.neuron.connect(other.neuron)
            self.neuron.weights_history.clear()
            
    @staticmethod
    def add(neuron):
        Neurons.neurons.append(neuron)
        if neuron.race == Neurons.INPUT:
            Neurons.input_neurons.append(neuron)
            
        if neuron.race== Neurons.OUTPUT:
            Neurons.output_neurons.append(neuron)
       
    @staticmethod
    def change(pos, value):
        n = Neurons.pressed(pos)
        if n != False:
            if n.race == Neurons.INPUT:
                n.neuron.set_value(value)
            else:
                n.neuron.bias = value
        else:
            main, previous = Neurons.weight_pressed(pos)
            if main != False:
                main.neuron.weights[main.neuron.neurons.index(previous.neuron)] = value
                                 
    @staticmethod
    def reset(pos):
        pressed_Neuron = Neurons.pressed(pos)
        if pressed_Neuron != False:
            pressed_Neuron.neuron.reset()
        else:
            main, previous = Neurons.weight_pressed(pos)
            if main != False:
                main.neuron.reset(previous.neuron)
                
    @staticmethod
    def delete(pos):
        n = Neurons.pressed(pos)
        
        if n != False:
            Neurons.disconnect_all(pos)
            Neurons.neurons.remove(n)
            if n in Neurons.input_neurons: Neurons.input_neurons.remove(n)
            if n in Neurons.output_neurons: Neurons.output_neurons.remove(n)
            del(n.neuron)
            del(n)
        else:
            main, previous = Neurons.weight_pressed(pos)
            if main != False:
                main.neuron.disconnect(previous.neuron)
                main.previous_neurons.remove(previous)
    
    @staticmethod
    def draw(screen, debug_mode = False):
        # Draw connections
        for n in Neurons.neurons:
            for idx, other in enumerate(n.previous_neurons):
                color = GREEN if n.neuron.weights[idx] > 0 else RED
                pygame.draw.line(screen, color, other.pos, n.pos, round(smooth_increase(n.neuron.weights[idx])))
                if debug_mode:
                    text_x = (other.pos[0] - n.pos[0]) * Neurons.WEIGHT_RATIO + n.pos[0] 
                    text_y = (other.pos[1] - n.pos[1]) * Neurons.WEIGHT_RATIO + n.pos[1]
                    text = WEIGHT_FONT.render(str(round(n.neuron.weights[idx], 3)), 1, WHITE)
                    screen.blit(text, (text_x, text_y))
        
        # Draw neuron
        for n in Neurons.neurons:
            pygame.draw.circle(screen, n.color, n.pos, n.radius)
            if debug_mode:
                if n.race == Neurons.INPUT: val = str(round(n.neuron.value, 3))
                elif n.race == Neurons.OUTPUT: val = str(round(n.neuron.bias, 2)) +"|"+  str(round(n.neuron.value, 2))
                else: val = str(round(n.neuron.bias, 3))
                text = BIAS_FONT.render(val, 1, WHITE)
                screen.blit(text, (n.rect.center[0]- text.get_width()//2, n.rect.center[1] - text.get_height()//2))
            
    
    @staticmethod
    def backprop(train_x, train_y):
        for i, x in enumerate(train_x):
            Neurons.input_neurons[i].neuron.set_value(x)
            
        for i, y in enumerate(train_y):
            Neurons.output_neurons[i].neuron.quadratic_cost(y)
            
            
    @staticmethod
    def pressed(pos):
        for n in Neurons.neurons:
            if n.rect.collidepoint(pos):
                return n
        return False
    
    @staticmethod
    def info(pos, manager):
        n = Neurons.pressed(pos)
        if n != False and n.race != Neurons.INPUT:
            InfoWindow(n, manager, Neurons.neurons.index(n))
            
    @staticmethod
    def process_events(event):
        InfoWindow.process_events(event)
                
                
# =============================================================================
# Auxiliary Methods  
# =============================================================================
    @staticmethod
    def weight_pressed(pos):
        rect = 0
        for n  in Neurons.neurons:
            for i, other in enumerate(n.previous_neurons):
                x = (other.pos[0] - n.pos[0]) * Neurons.WEIGHT_RATIO + n.pos[0] 
                y = (other.pos[1] - n.pos[1]) * Neurons.WEIGHT_RATIO + n.pos[1]
                rect = pygame.Rect(x, y, WEIGHT_COL, WEIGHT_COL)
                
                if rect.collidepoint(pos):
                    return (n, other)
                
        return (False, False)

    @staticmethod
    def disconnect_all(neuron):
        n = Neurons.pressed(neuron)
            
        if n != False:
            for prev_n in reversed(n.neuron.neurons):
                n.neuron.disconnect(prev_n)
            n.previous_neurons.clear()
            
            for next_n in reversed(n.neuron.next_neurons):
                next_n.disconnect(n.neuron)
                
            for other in reversed(Neurons.neurons):
                if n in other.previous_neurons:
                    other.previous_neurons.remove(n)     
            
# =============================================================================
# Test Methods        
# =============================================================================
    @staticmethod
    def simple_test(text_x, test_y):
        error = 0
        for i, x in enumerate(text_x):
            for j, inpt in enumerate(x):
                Neurons.input_neurons[j].neuron.set_value(inpt)
            for j, outpt in enumerate(test_y[i]):
                error += 0.5 * (Neurons.output_neurons[j].neuron.value - outpt)**2
        
        error /= len(text_x)
        print("Error:", error)
        
    @staticmethod
    def graph_test():
        for i in range (Neurons.RESOLUTION):
            for j in range (Neurons.RESOLUTION):
                Neurons.input_neurons[0].neuron.set_value(i / (Neurons.RESOLUTION- 1))
                Neurons.input_neurons[1].neuron.set_value(j / (Neurons.RESOLUTION - 1))
                for n in Neurons.neurons:
                    n.values_plot[i][j] = n.neuron.value