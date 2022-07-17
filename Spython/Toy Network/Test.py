import random
import Neurons
import Networks
import numpy as np
import math
import matplotlib.pyplot as plt
import pygame as pg
import Neurons2

# a simple pygame text example
# vegaseat


pg.init()

# use a (r, g, b) tuple for color
yellow = (255, 255, 0)

# create the basic window/screen and a title/caption
# default is a black background
screen = pg.display.set_mode((640, 280))
pg.display.set_caption("Text adventures with Pygame")
# pick a font you have and set its size
myfont = pg.font.SysFont("Comic Sans MS", 30)
# apply it to text on a label
label = myfont.render("Python and Pygame are Fun!", 1, yellow)
# put the label object on the screen at point x=100, y=100
screen.blit(label, (100, 100))
# show the whole thing
pg.display.flip()

# event loop
while True:
    for event in pg.event.get():
        # exit conditions --> windows titlebar x click
        if event.type == pg.QUIT:
            raise SystemExit