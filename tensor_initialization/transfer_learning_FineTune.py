import sys

import  torch
import torch.nn as nn #All neural network modules, nn.Linear,nn.Conv2d, BatchNorm, Loss function
import torch.optim as optim #For all Optimization algorithm,SGD, Adam,etc.
import torch.nn.functional as F #All functions that don't have any parameters
import torchvision.models
from torch.utils.data import DataLoader #Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets  #Has standard datasets we can import in a nice way
import torchvision.transforms as transforms #Transformations we can perform on our dataset

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
sys.exit()
#Hyperparameter
in_channels = 1
num_classes = 10

learning_rate = 0.001
batch_size = 1024
num_epochs  =5

class Identify(nn.Module):
    def __init__(self):
        super(Identify, self).__init__()


    def forward(self,x):
        return x
#load pretrain model and modify it
model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identify()
model.classifier = nn.Sequential(nn.Linear(512,100),
                                 nn.ReLU(),
                                 nn.Linear(100,10))

model.to(device)

#Load data

train_dataset = datasets.CIFAR10(root='dataset/',train = True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset = train_dataset,batch_size=batch_size, shuffle=True)


#Initialize network

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
#Train Network

for epochs in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        #forward
        scores = model(data)
        loss = criterion(scores,targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step
        optimizer.step()



#Check accuracy on training & test to use how good our model
def check_accuracy(loader,model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuacry on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct +=(predictions ==y).sum()
            num_samples+=predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy  {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()


check_accuracy(train_loader,model)
check_accuracy(test_loader,model)




