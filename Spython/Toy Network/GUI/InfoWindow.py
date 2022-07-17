import matplotlib.pyplot as plt
from Utils import NUMBERS
import pygame_gui as pygui
import pygame
import os

STUPID_W, STUPID_H = 25 ,52
WIN_W, WIN_H = 220, 300
IMAGE_W, IMAGE_H = 100, 100
INP_W, INP_H = 138, 22

TEXT_W, TEXT_H = WIN_W - STUPID_W, WIN_H - IMAGE_H - INP_H -STUPID_H
BUT_W, BUT_H = WIN_W - IMAGE_W - STUPID_W, IMAGE_H // 2

GRAPH_W, GRAPH_H = 334, 195
GRAPH_WIN_W, GRAPH_WIN_H = GRAPH_W + STUPID_W + 4, GRAPH_H + STUPID_H + 3

RESOLUTION = 40
TEXT1 = '<font face=’comicsans’ size=3.5>Delta = {}<br>Value = {}<br>Weights = {}</font>'
TEXT2 = '<font face=’comicsans’ size=3.5>Learning Rate =</font>'
class InfoWindow():
    
    windows = []
    
    def __init__(self, ui_neuron, manager, num = ""):
        self.neuron = ui_neuron.neuron
        self.manager = manager
        InfoWindow.windows.append(self)
        
        # Create Window
        window_rect = pygame.Rect(ui_neuron.pos[0], ui_neuron.pos[1], WIN_W, WIN_H)
        self.title = ui_neuron.race +" " +str(num)
        self.window = pygui.elements.UIWindow(window_rect, manager, self.title, resizable=True)
        
        # Create buttons
        but1_rect = pygame.Rect(IMAGE_W - 3, - 3, BUT_W, BUT_H + 5)
        but2_rect = pygame.Rect(IMAGE_W - 3, BUT_H - 3, BUT_W, BUT_H + 5)
        self.but1 = pygui.elements.UIButton(but1_rect, "Weights", manager, container=self.window)
        self.but2 = pygui.elements.UIButton(but2_rect, "Deltas", manager, container=self.window)
               
        # Create image
        filename = os.path.join("Graphs", self.title + ".jpg")
        
        plt.imsave(filename, ui_neuron.values_plot, cmap = plt.cm.Blues)
        image_rect = pygame.Rect(0, 0, IMAGE_W, IMAGE_H)
        image = pygame.transform.scale(pygame.image.load(filename), (IMAGE_W, IMAGE_H))
        value_plot = pygui.elements.UIImage(image_rect, image , manager, container=self.window)
        
        # Create input field
        des_rect = pygame.Rect(-3, IMAGE_H-2, INP_W, INP_H)
        inp_rect = pygame.Rect(INP_W-10, IMAGE_H-2, WIN_W - INP_W, INP_H)
        description = pygui.elements.UITextEntryLine(des_rect, manager, container=self.window)
        self.inp_field = pygui.elements.UITextEntryLine(inp_rect, manager, container=self.window)
        description.enable()
        description.set_text("Learning rate =")
        self.inp_field.set_text(str(self.neuron.learning_rate))
        self.inp_field.set_allowed_characters(NUMBERS)
        
        # Create text box
        text_rect = pygame.Rect(-3, IMAGE_H+INP_H, TEXT_W, TEXT_H)
        text = TEXT1.format(round(self.neuron.deltas_history[-1], 3),
                            round(self.neuron.value, 3),[round(w, 3) for w in self.neuron.weights])
        text_box = pygui.elements.UITextBox(text, text_rect, manager, container=self.window)
        
        
        
        
    def weights_window(self):
        filename = os.path.join("Graphs", "W" + self.title + ".jpg")
        
        plt.plot(self.neuron.weights_history)
        plt.tick_params(axis = 'both', bottom=False, labelbottom=False, top=False, labeltop=False)
        plt.savefig(filename ,bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()
        
        window_rect = pygame.Rect(self.but1.rect.topleft[0], self.but1.rect.topleft[1],GRAPH_WIN_W, GRAPH_WIN_H)
        window = pygui.elements.UIWindow(window_rect, self.manager, "Weights history: " + self.title, resizable=True)
        
        image_rect = pygame.Rect(0, 0, GRAPH_W, GRAPH_H)
        image = pygame.transform.scale(pygame.image.load(filename), (GRAPH_W, GRAPH_H))
        weights_plot = pygui.elements.UIImage(image_rect, image , self.manager, container=window)
        
        
        
    def deltas_window(self):
        filename = os.path.join("Graphs", "D" + self.title + ".jpg")
        
        plt.plot(self.neuron.deltas_history)
        plt.tick_params(axis = 'both', bottom=False, labelbottom=False, top=False, labeltop=False)
        plt.savefig(filename ,bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()
        
        window_rect = pygame.Rect(self.but1.rect.topleft[0], self.but1.rect.topleft[1],GRAPH_WIN_W, GRAPH_WIN_H)
        window = pygui.elements.UIWindow(window_rect, self.manager, "Delta history: " + self.title, resizable=True)
        
        image_rect = pygame.Rect(0, 0, GRAPH_W, GRAPH_H)
        image = pygame.transform.scale(pygame.image.load(filename), (GRAPH_W, GRAPH_H))
        weights_plot = pygui.elements.UIImage(image_rect, image , self.manager, container=window)
    
    
    @staticmethod
    def process_events(event):
        if event.type == pygame.USEREVENT:
            if event.user_type == pygui.UI_TEXT_ENTRY_CHANGED:
                for w in InfoWindow.windows:
                    if event.ui_element == w.inp_field:
                        if w.inp_field.get_text() == "":
                            w.neuron.learning_rate = 0
                            w.inp_field.set_text("0")
                        else:
                            w.neuron.learning_rate = float(w.inp_field.get_text())
            if event.user_type == pygui.UI_BUTTON_PRESSED:
                for w in InfoWindow.windows:
                    if event.ui_element == w.but1:
                        w.weights_window()
                    if event.ui_element == w.but2:
                        w.deltas_window()
                    if event.ui_element == w.window.close_window_button:
                        InfoWindow.windows.remove(w)
                        del(w)
                        