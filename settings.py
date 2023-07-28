import pygame

# Colors
RED = pygame.Color(255,0,0)
GREEN = pygame.Color(0,255,0)
BLUE = pygame.Color(0,0,255)
BLACK = pygame.Color(0,0,0)
WHITE = pygame.Color(255,255,255)
GRAY = pygame.Color(50,50,50)
YELLOW = pygame.Color(255,255,0)


# Dimensions
WINDOW_DIM = (28*18,28*18)
BAR_DIM = (28*18,56)
GRID_DIM = (28,28)
PIXEL_SIZE = (WINDOW_DIM[0] // GRID_DIM[0], (WINDOW_DIM[1] - BAR_DIM[1]) // GRID_DIM[1])
BTN_DIMS = (30,30)
BTN_Y_CENTER = WINDOW_DIM[1] - BAR_DIM[1]/2 - BTN_DIMS[1]/2

# RATE
FPS = 60

class Button():
    def __init__(self, x, y, width, height, color=BLACK, text=None) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.text = text

    def draw(self, window):
        pygame.draw.rect(window, color=self.color, rect=(self.x, self.y, self.width, self.height))

    def clicked(self):
        x_pos, y_pos = pygame.mouse.get_pos()
        x_condition = (x_pos >= self.x) and (x_pos <= self.x + self.width)
        y_condition = (y_pos >= self.y) and (y_pos <= self.y + self.height)

        return x_condition and y_condition

buttons = [
    Button(x=10, y=BTN_Y_CENTER, width=BTN_DIMS[0], height=BTN_DIMS[1], color=RED),
    Button(x=28*18 - 10 - BTN_DIMS[0], y=BTN_Y_CENTER, width=BTN_DIMS[0], height=BTN_DIMS[1], color=GREEN, text="GET DATA"),
    Button(x=28*18 - 20 - 2*BTN_DIMS[0], y=BTN_Y_CENTER, width=BTN_DIMS[0], height=BTN_DIMS[1], color=BLUE, text="SAVE"),
    Button(x=20 + BTN_DIMS[0], y=BTN_Y_CENTER, width=BTN_DIMS[0], height=BTN_DIMS[1], color=YELLOW)
]



        

