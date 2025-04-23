import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import config
import tqdm 
from torchmetrics import Accuracy




class ConvBlock(nn.Module):
    def __init__(self, in_channels,out_Channels,Bnorm = False):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_channels,out_channels=out_Channels,kernel_size=(3,3),padding='valid'),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2,stride=2)]
        if Bnorm:
            layers.append(nn.BatchNorm2d(out_Channels))
        self.BackBone = nn.Sequential(*layers)

    def forward(self,x):
        return self.BackBone(x)
    

class AnimalModel(nn.Module):
    def __init__(self,num_classes:int = 6):
        super(AnimalModel,self).__init__()
        self.BackBone = nn.Sequential(ConvBlock(3,16,True),
                            ConvBlock(16,32,True),
                            ConvBlock(32,64,True),
                            nn.Flatten(),
                            nn.Linear(43264,out_features=num_classes),
                            nn.Softmax(dim=1))
        self.optimizer = optim.Adam(self.parameters(),lr=config.learningRate)
    def forward(self,x):
        return self.BackBone(x)  


@torch.no_grad()
def plotConfMatrix(model,batch,device,writer,epoch):
    images,labels = batch
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    preds = torch.argmax(outputs,dim=1)
    conf_matrix = torch.zeros((6,6))
    for i in range(len(labels)):
        conf_matrix[labels[i]][preds[i]] += 1
    writer.add_image("Confusion Matrix",conf_matrix,epoch)

class Trainer():
    def __init__(self,criterion,device,writer):
        self.criterion = criterion
        self.device = device
        self.writer = writer
    
    def step(self,model,batch,train = True):
        images,labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        outputs = model(images)
        loss = self.criterion(outputs,labels)
        if train:
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
        return loss.item(),outputs,labels
    
    def train_epoch(self,model,loader,epoch,train = True):
        msg = "Training" if train else "Validation"
        model.train() if train else model.eval()
        pbar = tqdm.tqdm(loader, desc=f"{msg} Epoch {epoch+1}/{config.num_epochs}", unit="batch")
        eloss,eacc = 0.0,0.0
        metric = Accuracy(task='multiclass',num_classes=6).to(self.device)
        for batch in pbar:
            loss,outputs,labels = self.step(model,batch,train)
            eloss += loss
            acc = metric(outputs,labels)
            eacc += acc.item()
            pbar.set_postfix({"loss": eloss / (pbar.n + 1), "acc": eacc / (pbar.n + 1)})
        eloss /= len(loader)
        eacc /= len(loader)
        self.writer.add_scalar(f"{msg}/loss",eloss,epoch)
        self.writer.add_scalar(f"{msg}/acc",eacc,epoch)
    
    def train(self,model,train_loader,num_epochs,val_loader=None):
        for epoch in range(num_epochs):
            self.train_epoch(model,train_loader,epoch,train=True)
            if val_loader:
                self.train_epoch(model,val_loader,epoch,train=False)
            #if (epoch+1) % 5 == 0:
            #    plotConfMatrix(model,val_loader.dataset[0],self.device,self.writer,epoch)