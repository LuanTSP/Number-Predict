from mydata_training import MyNN, load_model
import torch
from torch.nn.functional import softmax
import cv2
from random import randint
import os


def get_samples():
    imgs = []
    targets = []
    for _ in range(10):
        folder = os.listdir("./MyData/")
        k = randint(0, len(folder) - 1)
        folder = folder[k]
        targets.append(int(folder))
        
        file = os.listdir(os.path.join("./MyData", folder))
        file = file[randint(0, len(file) - 1)]
        path = f"./MyData/{folder}/{file}"
        img = cv2.imread(path)
        img = torch.from_numpy(img).mean(dim=2, dtype=torch.float).reshape((1,28,28))
        imgs.append(img)
    return torch.stack(imgs), targets


def evaluate(model, imgs, target):
    output = model(imgs)
    pred = output.argmax(dim=1)

    print(f"Predicted: {pred}")
    print(f"Actual   : {torch.tensor(target)}")

if __name__ == "__main__":
    model = MyNN()
    model = load_model(model=model, path="./model_parameters/nn_mydata.pth.tar")
    model.eval()

    imgs, target = get_samples()
    evaluate(model=model, imgs=imgs, target=target)


