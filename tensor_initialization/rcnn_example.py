import  torch
import torch.nn as nn #All neural network modules, nn.Linear,nn.Conv2d, BatchNorm, Loss function
import torch.optim as optim #For all Optimization algorithm,SGD, Adam,etc.
import torch.nn.functional as F #All functions that don't have any parameters
from torch.utils.data import DataLoader #Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets  #Has standard datasets we can import in a nice way
import torchvision.transforms as transforms #Transformations we can perform on our dataset



#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs  =1


#Create RCNN
class RCNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layer,num_class):
        super(RCNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layer
        self.rnn = nn.RNN(input_size,hidden_size,num_layer,batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length,num_class)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(device)

        #Forward Prop
        out, _=self.rnn(x,h0)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out


#Create Fully Connected Network
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x  =F.relu(self.fc1(x))
        x  =self.fc2(x)
        return x

#Load data

train_dataset = datasets.MNIST(root='dataset/',train = True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset = train_dataset,batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/',train = False,transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset = test_dataset,batch_size=batch_size, shuffle=True)

#Initialize network
model = RCNN(input_size,hidden_size,num_layers,num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
#Train Network

for epochs in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):
        #Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
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
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)


            scores = model(x)
            _,predictions = scores.max(1)
            num_correct +=(predictions ==y).sum()
            num_samples+=predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy  {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()


check_accuracy(train_loader,model)
check_accuracy(test_loader,model)



