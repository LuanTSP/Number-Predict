import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
from sklearn.model_selection import train_test_split


class MyData(Dataset):
    def __init__(self, train=True) -> None:
        self.data = []
        self.labels = []
        # Get Data from files
        for directory in os.listdir("./MyData"):
            for file in os.listdir(os.path.join("./MyData", directory)):
                img = cv2.imread(os.path.join("./MyData", directory, file))
                img = torch.from_numpy(img).mean(dim=2, dtype=torch.float).reshape((1,28,28))
                label = int(directory)
                self.data.append(img)
                self.labels.append(label)
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, train_size=0.7, shuffle=True)
        if train:
            self.data = X_train
            self.labels = y_train
        else:
            self.data = X_test
            self.labels = y_test
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=32)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=64)
        self.batch_norm_3 = nn.BatchNorm2d(num_features=128)
        self.batch_norm_4 = nn.BatchNorm2d(num_features=256)
        self.dropout = nn.Dropout2d(p=0.25)
        self.relu = nn.functional.relu
        self.l1 = nn.Linear(in_features=256*18*18, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=128)
        self.l3 = nn.Linear(in_features=128, out_features=64)
        self.l4 = nn.Linear(in_features=64, out_features=10)
        self.softmax = nn.functional.softmax
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm_1(x)
        x = self.conv2(x)
        x = self.batch_norm_2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.batch_norm_3(x)
        x = self.conv4(x)
        x = self.batch_norm_4(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0],-1))
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.softmax(self.l4(x), dim=0)
        return x



def train_one_epoch(model, dataloader, optimizer, loss_fn):
    epoch_loss = []
    correct = 0
    for step, (imgs, labels) in enumerate(dataloader):

        # zero grads
        optimizer.zero_grad()
        
        # predict
        outputs = model(imgs)

        # calculate accuraccy
        preds = outputs.argmax(dim=1)
        for pred, label in zip(preds, labels):
            if label.item() == pred.item():
                correct += 1

        # loss
        loss = loss_fn(outputs, labels)
        epoch_loss.append(loss.item())

        # calculate weight parameters
        loss.backward()

        # optimizer step
        optimizer.step()
    accur = correct / len(train_dataset)
    return epoch_loss, accur


def train(epochs, model, dataloader, optimizer, loss_fn):
    total_loss = []
    accur_hist = []
    best_loss = 10**4
    for epoch in range(epochs):
        epoch_loss, accur = train_one_epoch(model=model, dataloader=dataloader, optimizer=optimizer, loss_fn=loss_fn)
        total_loss.append(epoch_loss)
        accur_hist.append(accur)
        mean_loss = torch.tensor(epoch_loss).mean(dim=0).item()
        
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_model(path=path)
            model = load_model(model=model, path=path)

        print("-"*50)
        print(f"Epoch: [{epoch + 1}/{epochs}], Loss: [{np.array(epoch_loss).mean():.5f}] - Accuracy: [{accur:.5f}] - Best Loss: [{best_loss:.5f}]")
    return total_loss, accur_hist


def make_graphs(total_loss, accur):
    x = np.linspace(start=1, stop=epochs, num=epochs)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    axes[0].plot(x, np.array(total_loss).mean(axis=1), "--o", c='red')
    axes[0].set_title("Mean Cross Entropy Loss x Epochs")
    axes[0].set_ylabel("Mean Cross Entropy Loss")
    axes[0].set_xlabel("Epochs")

    axes[1].plot(x, accur, "--o", c='red')
    axes[1].set_title("Train Accuracy x Epochs")
    axes[1].set_ylabel("Train Accuracy")
    axes[1].set_xlabel("Epochs")


def save_model(path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    try:
        model.load_state_dict(torch.load(path))
    except:
        print(f"Model Parameters Not Found at [{path}]")
        pass
    finally:
        return model


def evaluate_test(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for step, data in enumerate(test_dataloader):
            imgs, labels = data

            # Make predictions
            outputs = model(imgs)
            
            # calculate accuraccy
            preds = outputs.argmax(dim=1)
            for pred, label in zip(preds, labels):
                if label.item() == pred.item():
                    correct += 1
    
    return correct / len(test_dataset)

if __name__ == "__main__":
    # PARAMS
    batch_size = 50
    epochs = 5
    learning_rate = 1e-4
    path = "./model_parameters/nn_mydata.pth.tar"

    # Creating Dataset and Dataloader
    model = MyNN()
    model = load_model(model=model, path=path) # Load Model
    train_dataset = MyData(train=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MyData(train=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Setting optimizer and loss_fn
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # asserting shape
    imgs, labels = next(iter(train_dataloader))
    output = model(imgs)
    preds = output.argmax(dim=1)
    assert preds.shape == labels.shape, "Output Shape Error"

    # Train Model
    total_loss, accur = train(epochs=epochs, model=model, dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn)
    
    # Evaluate model in Test Data
    print(f"\nTest Accuracy: {evaluate_test(model=model):.5f}")
    
    save_model(path=path)

    # make_graphs(total_loss=total_loss, accur=accur)
    plt.show()

