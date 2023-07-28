import pygame
from settings import *
from math import floor
import numpy as np
import matplotlib
from mydata_training import MyNN, load_model
import cv2
import torch
import uuid

def get_grid_index():
    x, y = pygame.mouse.get_pos()
    x, y = x // PIXEL_SIZE[0], y // PIXEL_SIZE[1]
    return x, y

def draw_grid_lines(window):
    for x in range(0, WINDOW_DIM[0] + 1, PIXEL_SIZE[0]):
        pygame.draw.line(window, GRAY, (x, 0), (x, WINDOW_DIM[1] - BAR_DIM[1]))

    for y in range(0, WINDOW_DIM[1] - BAR_DIM[1] + 1, PIXEL_SIZE[1]):
        pygame.draw.line(window, GRAY, (0, y), (WINDOW_DIM[0] ,y))

def draw_grid(window, grid):
    for i, row in enumerate(grid):
        for j, color in enumerate(row):
            pygame.draw.rect(window, color, (i * PIXEL_SIZE[0], j * PIXEL_SIZE[1], PIXEL_SIZE[0], PIXEL_SIZE[1]))   

def darken(color: pygame.Color, amount: int):
    return pygame.Color(color.r - amount, color.g - amount, color.b - amount)
    
def difuse_color(grid, x, y):
    r, g, b = 0, 0 ,0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            try:
                r += grid[x+i][y+j].r
                g += grid[x+i][y+j].g
                b += grid[x+i][y+j].b
            except:
                raise IndexError
    r = floor(r / 9)
    g = floor(g / 9)
    b = floor(b / 9)
    return pygame.Color(r, g, b)

def paint(grid, drawing_color):
    x, y = get_grid_index()
    # Set color of principal pixel
    try:
        grid[x][y] = drawing_color
    except:
        raise IndexError
    # Set color of adjacent pixels
    mean_color = difuse_color(grid=grid, x=x, y=y)
    for i in [-1, 1]:
        try:
            if grid[x+i][y] != drawing_color:
                grid[x+i][y] = mean_color
        except:
            raise IndexError
    for j in [-1, 1]:
        try:
            if grid[x][y+j] != drawing_color:
                grid[x][y+j] = mean_color
        except:
            raise IndexError
        
    
def init_grid():
    grid = []
    for x in range(GRID_DIM[0]):
        row = []
        for y in range(GRID_DIM[1]):
            row.append(BLACK)
        grid.append(row)
    return grid


def grid_to_gray(grid):
    gray_img = []
    for x in range(GRID_DIM[0]):
        row = []
        for y in range(GRID_DIM[1]):
            gray = 0.229*grid[x][y].r + 0.587*grid[x][y].g + 0.114*grid[x][y].b
            row.append(floor(gray))
        gray_img.append(row)
    return np.array(gray_img).T


def collect_data(n):
    img = grid_to_gray(grid=grid)
    matplotlib.image.imsave(f"./MyData/{n}/{n}.{uuid.uuid4()}.png", img)


def predict(model):
    img = grid_to_gray(grid=grid)
    matplotlib.image.imsave("./predicted.png", img)
    
    img = cv2.imread("./predicted.png")
    img = torch.from_numpy(img).mean(dim=2, dtype=torch.float).reshape((1,28,28))
    
    img2 = cv2.imread("./base.png")
    img2 = torch.from_numpy(img2).mean(dim=2, dtype=torch.float).reshape((1,28,28))

    imgs = torch.stack([img, img2])
    
    output = model(imgs)
    preds = output.argmax(dim=1)
    preds[1] = preds[0] + 1
    return preds[0].item()


pygame.init()
running = True
clock = pygame.time.Clock()
window = pygame.display.set_mode((WINDOW_DIM[0],WINDOW_DIM[1]))
pygame.display.set_caption("Draw a Number")
grid = init_grid()
drawing_color = WHITE
can_collect = False
data_number = 0
model = MyNN()
model = load_model(model=model, path="./model_parameters/nn_mydata.pth.tar")
model.eval()

while running:
    clock.tick(FPS)
    window.fill(GRAY)
    # Draw Grid
    draw_grid(window=window, grid=grid)

    # Draw gridlines
    draw_grid_lines(window=window)

    # Draw buttons
    for btn in buttons:
        btn.draw(window=window)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if pygame.mouse.get_pressed()[0]:
            try:
                # Try paint
                paint(grid=grid, drawing_color=drawing_color)
            except IndexError:
                # check buttons

                # Delete Button
                if buttons[0].clicked():
                    grid = init_grid()
                # Initiate Collect Button
                elif buttons[1].clicked():
                    can_collect = True
                    data_number = 0
                    pygame.display.set_caption(f"Collecting Data: {data_number}")
                # Save data Button
                elif buttons[2].clicked() and can_collect:
                    if (data_number) <= 9:
                        collect_data(data_number)
                        data_number += 1
                        pygame.display.set_caption(f"Collecting Data: {data_number}")
                        pygame.display.update()
                        pygame.time.wait(300)
                    else:
                        data_number = 0
                        can_collect = False
                        pygame.display.set_caption("Draw a Number")
                    grid = init_grid()
                # Predict Button
                elif buttons[3].clicked():
                    pred = predict(model=model)
                    pygame.display.set_caption(f"{pred}")
                    pygame.display.update()
    pygame.display.update()
pygame.quit()