from NeuronsUI import Neurons
from Utils import BLACK, RED, RADIUS, GRAY, snap, SNAP_WIDTH, SNAP_HEIGHT, NUMBERS
import random
import pygame
import pygame_gui as pygui
import data.Cross as data
pygame.init()
pygame.font.init()


WIDTH, HEIGHT = 800, 500
L_W = 100
B_W, B_H = L_W, 60
I_W, I_H = L_W, 20
WTF_IS_THIS, FPS = 1000, 60
CONT_FACTOR = 20
ACTIONS = [Neurons.SIGMOID]
ACTIONS.extend(['Connect', 'Set', 'Reset', 'Delete', 'Info', 'None'])

screen = pygame.display.set_mode([WIDTH,HEIGHT])
manager = pygui.UIManager((WIDTH, HEIGHT))
clock = pygame.time.Clock()
train_x = data.x_train
train_y = data.y_train

connecting = False
connecting_neuron = 0

debug_mode = True
cont_backprop = False

ACTION_LIST = pygui.elements.UISelectionList(pygame.Rect(WIDTH - L_W, -2, L_W + 2, HEIGHT + 4), ACTIONS, manager)
B_BACKPROP = pygui.elements.UIButton(pygame.Rect(WIDTH - B_W, HEIGHT - B_H,  B_W + 2, B_H + 2), "Backprop", manager)
I_ITERATIONS = pygui.elements.UITextEntryLine(pygame.Rect(WIDTH - I_W, HEIGHT-B_H-I_H, I_W+2, I_H+4), manager)
B_CONT_BACKPROP = pygui.elements.UIButton(pygame.Rect(WIDTH-B_W,HEIGHT-2*B_H-I_H,B_W+2,B_H+4), "Continuous", manager)
B_DEBUG = pygui.elements.UIButton(pygame.Rect(WIDTH - B_W, HEIGHT - 3*B_H - I_H,  B_W + 2, B_H + 4), "Debug", manager)
B_DEBUG_MODE = pygui.elements.UIButton(pygame.Rect(WIDTH - B_W, HEIGHT -4*B_H-I_H,B_W+2,B_H+4), "Debug Mode", manager)
I_SET = pygui.elements.UITextEntryLine(pygame.Rect(WIDTH - I_W, HEIGHT-4*B_H-I_H*2, I_W+2, I_H+4 ), manager)
I_ITERATIONS.set_allowed_characters('numbers')
I_ITERATIONS.set_text('1000')
I_SET.set_allowed_characters(NUMBERS)
I_SET.set_text('0')



def mouse_up(pos):
    s_pos = snap(pos)
    selection = ACTION_LIST.get_single_selection()
    
    if selection == "Connect":
        connect(s_pos)
        
    if selection == "Set":
        Neurons.change(pos, float(I_SET.get_text()))
        
    if selection == "Reset":
        Neurons.reset(pos)
    
    if selection == "Delete":
        Neurons.delete(pos)
        
    if selection == "Info":
        Neurons.info(s_pos, manager)
    
    # Create new neuron
    elif selection in Neurons.RACES.keys():
        valid_pos = s_pos[0]>0 and s_pos[0]<WIDTH-L_W and s_pos[1]>0 and s_pos[1]<HEIGHT
        if not Neurons.pressed(s_pos) and valid_pos:
            Neurons.add(Neurons(s_pos, RADIUS, selection))
         
def debug():
    print()
    print("--------------Debug--------------")
    for i, n in enumerate(Neurons.neurons):
        if n not in Neurons.input_neurons:
            print("Neuron =", i)
            print("----------")
            print("Weights", len(n.neuron.weights))
            print("Previous Neurons", len(n.neuron.neurons))
            print("Previous UI Neurons", len(n.previous_neurons))
            print("Next Neurons", len(n.neuron.next_neurons))
            print("Learning Rate", n.neuron.learning_rate)
            print()
        
    for i, n in enumerate(Neurons.input_neurons):
        print("Input Neuron =", i)
        print("----------------")
        print("Next Neurons", len(n.neuron.next_neurons))
        print("Value", n.neuron.value)
        print()
    
    print("Output Neurons", len(Neurons.output_neurons))
    print("All Neurons", Neurons.neurons)
        
def connect(pos):
    global connecting, connecting_neuron
    pressed_neuron = Neurons.pressed(pos)
    
    if pressed_neuron != False:
        if not connecting:
            connecting_neuron = pressed_neuron
        else :
            pressed_neuron.connect(connecting_neuron)
            
        connecting = not connecting
        
    elif connecting : connecting = False  

def draw_lines():
    for i in range(0, WIDTH, SNAP_WIDTH):
        pygame.draw.line(screen, GRAY, (i, 0), (i, HEIGHT))
        
    for i in range(0, HEIGHT, SNAP_HEIGHT):
        pygame.draw.line(screen, GRAY, (0, i), (WIDTH, i))
        
def draw_connection():
    if connecting:
        pygame.draw.line(screen, RED, connecting_neuron.pos, pygame.mouse.get_pos())
        
def list_selection(event):
    if event.ui_element == ACTION_LIST:
        pass

def button_press(event):
    global debug_mode, cont_backprop
    if event.ui_element == B_DEBUG:
        debug()
    
    if event.ui_element == B_BACKPROP:
        backprop(int(I_ITERATIONS.get_text()))
        
    if event.ui_element == B_DEBUG_MODE:
        debug_mode = not debug_mode
        
    if event.ui_element == B_CONT_BACKPROP:
        cont_backprop = not cont_backprop

def update():
    screen.fill(BLACK)
    draw_lines()
    draw_connection()
    Neurons.draw(screen, debug_mode)
    manager.draw_ui(screen)
    pygame.display.flip()
    if cont_backprop:
        backprop(CONT_FACTOR)
 
        
def backprop(num = 1):
    for i in range(num):
        r = random.randint(0, len(train_y) - 1)
        Neurons.backprop(train_x[r], train_y[r])
    Neurons.graph_test()

def read_data():
    for i in range(data.inputs):
        pos = snap((80, 200 + i * 100))
        Neurons.add(Neurons(pos , RADIUS, Neurons.INPUT))
        
    for j in range(data.outputs):
        pos = snap((WIDTH - L_W * 2, 240 + j * 100))
        Neurons.add(Neurons(pos, RADIUS, Neurons.OUTPUT))
        

# =============================================================================
# Main Game
# =============================================================================
def run():
    read_data()
    running = True
    while running:
        delta_time = clock.tick(FPS) / WTF_IS_THIS
        manager.update(delta_time)
        for event in pygame.event.get():
            manager.process_events(event)
            Neurons.process_events(event)
            if event.type == pygame.QUIT:
                pygame.quit()
            
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_up(event.pos)
            
            if event.type == pygame.USEREVENT:
                if event.user_type == pygui.UI_BUTTON_PRESSED:
                    button_press(event)
                    
            if event.type == pygame.USEREVENT:
                if event.user_type == pygui.UI_SELECTION_LIST_NEW_SELECTION:
                    list_selection(event)       
        
        update()
    
run()
        