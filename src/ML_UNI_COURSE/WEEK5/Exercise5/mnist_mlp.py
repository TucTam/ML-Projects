import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as ff
from torchvision import datasets
import torchvision.transforms as transforms
import torchmetrics

# Set device to use gpu for faster training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Transformer for the image data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])

# Downloading the mnist data
trainset = datasets.MNIST(root="../", train=True, transform=transform, download=False)
testset = datasets.MNIST(root="../", train=False, transform=transform, download=False)

# Split the downloaded dataset into features and labels(onehotencoded)
# and returns them as a numpy array
def feature_label_split(dataset):
    xdata = np.zeros((len(dataset), 28*28))
    ydata = np.zeros((len(dataset), 10))  
    for i, (x,y) in enumerate(dataset):
        xdata[i,:] = x.reshape(28*28)
        ydata[i,:] = np.asarray(ff.one_hot(torch.tensor(y), 10))   
    return np.asarray(xdata), np.asarray(ydata)
    
# Creating custom dataset class that can be used in the for initializing Dataloaders
class MNIST_Dataset(Dataset):
    def __init__(self, features, labels, transform):
        self.feats = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self, index):
        image = self.feats[index].reshape(28,28).astype("float32")
        label = self.labels[index].astype("float32")
        label = torch.Tensor(label)
        if self.transform:
            image = self.transform(image)
          
        return image, label

# Creating the model arcitechture
class MLPModel(nn.Module):
    def __init__(self, input_sz, hid1_sz, hid2_sz, hid3_sz, class_sz):
        super(MLPModel, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_sz, hid1_sz), # Input layer to hidden layer
            nn.ReLU(),
            nn.Linear(hid1_sz, hid2_sz),
            nn.ReLU(),
            nn.Linear(hid2_sz, hid3_sz),
            nn.ReLU(),            
            nn.Linear(hid3_sz, class_sz),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        output = self.sequence(data)
        return output


print("initializing the data and model")
# Hyperparams
Epochs = 10
batchsize = 100
inputsz = 28*28
hiddensize = 16
hiddensize2 = 16
hiddensize3 = 16
outputsize = 10
learning_rate = 0.0001

# Setting up the data
# Splitting into training and testing sets
xtrain, ytrain = feature_label_split(trainset)
xtest, ytest = feature_label_split(testset)

customtrainset = MNIST_Dataset(xtrain, ytrain, transform)
customtestset = MNIST_Dataset(xtest, ytest, transform)

trainloader = DataLoader(customtrainset, batch_size=batchsize, shuffle=True)
testloader = DataLoader(customtestset, batch_size=batchsize, shuffle=True)

# Setting up model, loss function and optimizer
model = MLPModel(input_sz=inputsz, hid1_sz=hiddensize, hid2_sz=hiddensize2, hid3_sz = hiddensize3, class_sz=outputsize).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

print("Training the model")
# Training the model
model.train()
loss = 0
loss_container = []
for epoch in range(Epochs):
    for image, label in trainloader:
        optimizer.zero_grad()
        image = image.reshape(-1, 28*28).to(device)
        label = label.to(device)
        
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    loss_container.append(loss)
        
print(f"MSELoss: {loss_container[-1]}")
    
# evaluating the model accuracy
Accuracymetric = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)
Accuracymetric2 = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)
model.eval()
with torch.no_grad():
    # evaluating model accuracy with testing data
    for image, label in testloader:
        image = image.reshape(-1,28*28).to(device)
        label = label.to(device)
        
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted = ff.one_hot(predicted, 10).to(device)
        
        Accuracymetric.update(predicted,label)

    # evaluating model accuracy with training data
    for image, label in trainloader:
        image = image.reshape(-1,28*28).to(device)
        label = label.to(device)
        
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted = ff.one_hot(predicted, 10).to(device)
        
        Accuracymetric2.update(predicted,label)
        
AccyracyScore = Accuracymetric.compute()
AccyracyScore2 = Accuracymetric2.compute()
print(f"Accuracy Score of testing data: {AccyracyScore:.4f}")        
print(f"Accuracy Score of training data: {AccyracyScore2:.4f}")

# Plotting the MSELoss during training
with torch.no_grad():
    for i in range(len(loss_container)):
        loss_container[i] = loss_container[i].to("cpu")
    plt.plot(np.linspace(1,Epochs, num=Epochs,endpoint=True), np.asarray(loss_container))
    plt.xlabel("epochs")
    plt.ylabel("MSEloss")
    plt.title("LOSS BY EPOCHS")    
    plt.show()