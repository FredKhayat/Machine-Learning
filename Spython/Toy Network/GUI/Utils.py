import pygame
pygame.font.init()

BIAS_FONT = pygame.font.SysFont('comicsans', 14)
WEIGHT_FONT = pygame.font.SysFont('comicsans', 17)

BLACK = (32, 34, 43)
ORANGE = (209, 145, 61)
GREEN = (62, 179, 46)
BLUE = (42, 52, 191)
GRAY = (58, 62, 79)
WHITE = (220,220,220)
RED = (232, 116, 116)

WEIGHT_COL = 15
NUMBERS = ['1','2','3','4','5','6','7','8','9','0','.', '-']

SNAP_WIDTH, SNAP_HEIGHT = 48, 48


RADIUS = 19

def snap(pos):
    return (SNAP_WIDTH * round(pos[0]/SNAP_WIDTH), SNAP_HEIGHT * round(pos[1]/SNAP_HEIGHT))

def smooth_increase(x):
    return max(-((1/(0.1*abs(x) + 0.05)) - 2) * 5, 1)