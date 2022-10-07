# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torchvision.transforms as transforms
import torchvision
import PIL
import os
from os import walk
import glob

tr = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])

print(os.listdir("train"))

test_set = torchvision.datasets.ImageFolder("test",transform=tr)
train_set = torchvision.datasets.ImageFolder("test",transform=tr)

batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size = batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size = batch_size , shuffle=True)

K= 2
print("Classes",K)

class CNN(nn.Module):
    def __init__(self,K):
        super(CNN,self).__init__()
        self.model = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2),    
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(128*7*7,512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512,K)   
    )
    
    def forward(self,X):
        out = self.model(X)
        return out

model = CNN(K)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def batch_gd(model,criterion,optimizer,train_loader,test_loader,epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    
    for it in range(epochs):
        model.train()
        d0=datetime.now()
        train_loss = []
        for inputs,targets in train_loader:
            inputs,targets = inputs.to(device),targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        
        model.eval()
        test_loss = []
        for inputs,targets in test_loader:
            inputs,targets = inputs.to(device),targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        
        dt = datetime.now()-d0
        print(f'Epoch {it+1}/{epochs}, Train loss: {train_loss:.4f},Test loss: {test_loss:.4f}, \
            Duration: {dt}')
    return train_losses,test_losses

train_losses,test_losses = batch_gd(model,criterion,optimizer,train_loader,test_loader,epochs=30)

plt.plot(train_losses,label='Train loss')
plt.plot(test_losses,label = 'Test loss')
plt.legend()
plt.show()

model.eval()
n_correct = 0.
n_total = 0.
for inputs,targets in train_loader:
    inputs,targets = inputs.to(device),targets.to(device)
    outputs = model(inputs)
    
    _,predictions = torch.max(outputs,1)
    n_correct +=(predictions==targets).sum().item()
    n_total+=targets.shape[0]
train_acc = n_correct/n_total

n_correct = 0.
n_total=0.

for inputs,targets in test_loader:
    inputs,targets = inputs.to(device),targets.to(device)
    outputs = model(inputs)
    
    _,predictions = torch.max(outputs,1)
    n_correct +=(predictions==targets).sum().item()
    n_total+=targets.shape[0]
test_acc = n_correct/n_total
print(f'Train acc: {train_acc*100:.4f}, Test acc: {test_acc*100:.4f}')

img = glob.glob("D:/neural networks/shoeclassifier/test/nike/Image_30.jpg")
for image in img:
    images = PIL.Image.open(image)
    trans = transforms.ToPILImage()
    trans1 = transforms.ToTensor()
    img_req = (trans1(images))
    plt.imshow(trans(trans1(images)))

type(img_req)
model.eval()
img_req.shape

tr

rgb_im = images.convert('RGB')

img = tr(rgb_im)
img = img.unsqueeze(dim=0)

img.shape

img = img.to(device)

item_labels = ['adidas','nike']

print(model(img))
max = torch.argmax(model(img))
print(f'Predicted image is {item_labels[max]}')
torch.save(model,'shoeclassifier.pt')
