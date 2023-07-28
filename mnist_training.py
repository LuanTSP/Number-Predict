import torch
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d, Dropout
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import relu, softmax
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from matplotlib import pyplot as plt
from tqdm import tqdm


class MnistDataset(Dataset):
    
    def __init__(self, transforms=None, train=True):
        self.dataset = MNIST(download=True, root="./", transform=transforms, train=train)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label


def show_data(dataset):
    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(10,5))
    for i, ax in enumerate(axes.flatten()):
        data, label = dataset[i]
        ax.imshow(data.reshape(28,28), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label)
    fig.suptitle("MNIST Dataset")
    plt.show()


class MnistNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool = MaxPool2d(kernel_size=2, stride=1)
        self.dropout = Dropout(p=0.2)
        self.l1 = Linear(in_features=64*22*22, out_features=128)
        self.l2 = Linear(in_features=128, out_features=128)
        self.l3 = Linear(in_features=128, out_features=10)
    

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(self.conv2(x))
        x = self.dropout(x)
        x = x.reshape((x.shape[0],-1))
        x = relu(self.l1(x))
        x = relu(self.l2(x))
        x = self.l3(x)
        return x


def print_state(epoch, accur, jump, loss):
    print("-"*30)
    if epoch % jump == 0:
        print(f"Epoch: [{epoch}/{epochs}] - Loss: [{loss:.5f}] - Accuracy: [{accur:.3f}]")
    else:
        print(f"Epoch: {epoch}")
    

def train_one_epoch(model, loss_fn, optimizer, dataloader):
    epoch_losses = []
    correct = 0
    for step, data in enumerate(dataloader):
        # Move Data to Device
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Make predictions
        outputs = model(imgs)
        
        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                correct += 1
        
        
        # Compute Loss and Gradients
        loss = loss_fn(outputs, labels)
        epoch_losses.append(loss.item())
        loss.backward()
        
        # Adjust Learning Weights
        optimizer.step()
    accur = correct/len(dataloader)
    return epoch_losses, accur


def train(model, loss_fn, optimizer, dataloader):
    print(f"Training on {device}:\n")
    total_loss = []
    for epoch in tqdm(range(1, epochs+1), desc="Progress"):
        epoch_losses, accur = train_one_epoch(model=model, loss_fn=loss_fn, optimizer=optimizer, dataloader=dataloader)
        total_loss.append(epoch_losses)
        print_state(epoch=epoch, accur=accur, jump=jump, loss=epoch_losses[-1])
    return total_loss


def plot_epoch_loss(epoch_losses):
    epochs_array = [i for i in range(1, len(epoch_losses) + 1)]

    plt.plot(epochs_array, epoch_losses, c='red', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Mean Cross Entropy Loss")
    plt.show()


def evaluate_one_epoch(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)

            # Make predictions
            outputs = model(imgs)
            
            # Calculate accuracy
            preds = outputs.argmax(dim=1)
            for i in range(len(preds)):
                if preds[i] == labels[i]:
                    correct += 1
    
    return correct / len(dataloader)


def load_model(model, path):
    try:
        model.load_state_dict(torch.load(path))
    except:
        pass
    finally:
        return model

if __name__ == "__main__":
    # PARAMS
    epochs = 10
    jump = 1
    batch_size = 100
    learning_rate = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "./model_parameters/nn_mnist_parameters.pth.tar"

    # Creating Datasets
    transforms = Compose([ToTensor()])
    train_dataset = MnistDataset(transforms=transforms, train=True)
    test_dataset = MnistDataset(transforms=transforms, train=False)

    # Show Data
    show_data(train_dataset)

    # Creating Dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Creating Model, optimizer and Loss Function
    model = MnistNN()
    model = load_model(model, checkpoint_path)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Assert
    for data in train_dataloader:
        imgs, labels = data
        output = model(imgs)
        assert output.shape == (batch_size, 10), "Shape Error"
        break

    # Training Model
    model.to(device)
    model.train()
    total_loss = train(model=model, loss_fn=loss_fn, optimizer=optimizer, dataloader=train_dataloader)
    print(f"Accuracy Test Dataset: {evaluate_one_epoch(model=model, dataloader=test_dataloader)}")

    # Save Model
    torch.save(model.state_dict(), checkpoint_path)

    # Plot Train Losses
    epoch_losses = torch.tensor(total_loss).mean(dim=1)
    plot_epoch_loss(epoch_losses=epoch_losses)
        
