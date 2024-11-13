import torch
import torch.nn as nn
from utils import imageSize
import tqdm
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x
    
class animalModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.flatten = nn.Flatten()
        modifier = imageSize//2**3
        self.fc1 = nn.Linear(512*modifier*modifier, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class convBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False, dropout=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None
        self.dropout = nn.Dropout(0.3) if dropout else None

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    

class Animal2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = convBlock2(in_channels, 64)
        self.conv2 = convBlock2(64, 64, pool=True,dropout=True)
        self.conv3 = convBlock2(64, 64, pool=True,dropout=True)
        self.conv4 = convBlock2(64, 64, pool=True,dropout=True)
        #self.conv5 = convBlock2(64, 64, pool=True,dropout=True)
        self.flatten = nn.Flatten()
        modifier = imageSize//2**3
        
        self.fc1 = nn.Linear(64*modifier*modifier, 512)
        self.act1 = nn.ReLU()
        self.bnorm = nn.BatchNorm1d(512)
        #self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        #self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.outp = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.bnorm(x)
        #x = self.dropout1(x)

        x = self.fc2(x)
        x = self.act2(x)
        #x = self.dropout2(x)

        x = self.fc3(x)
        x = self.act3(x)
        x = self.dropout3(x)
        x = self.outp(x)
        x = self.softmax(x)
        return x
    
    def validation_step(self,val_loader,criterion,device,accMetric,writer,epoch):
        valSteps = len(val_loader)
        val_accs = 0.0
        val_losses = 0.0
        for valStep, (images,labels) in enumerate(val_loader):
            with torch.no_grad():
                print(f'Validation rogeression is :{valStep*100/valSteps:.3f}%',end='\r')
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)
                preds = torch.argmax(outputs, dim=1)
                val_acc = accMetric(preds, labels)
                val_accs += val_acc
                val_loss = criterion(outputs, labels)
                val_losses += val_loss
            if writer is not None:
                writer.add_scalar('Accuracy/val_step', val_acc, epoch*valSteps+valStep)
                writer.add_scalar('Loss/val_step', val_loss, epoch*valSteps+valStep)
        if writer is not None:
            writer.add_scalar('Accuracy/val_epoch', val_accs/valSteps, epoch)
            writer.add_scalar('Loss/val_epoch', val_losses/valSteps, epoch)
        print(f'Validation Loss: {val_losses/valSteps} Validation Accuracy: {val_accs/valSteps}')


    def fit(self, train_loader, val_loader, epochs, optimizer, criterion,device,writer,accMetric):
        trainsteps = len(train_loader)
        for epoch in range(epochs):
            print('Epoch:', epoch)
            self.train()
            train_loss = 0.0
            train_accs = 0.0
            for trainingStep, (images, labels) in enumerate(train_loader):
                print(f'Progeression is :{trainingStep*100/trainsteps:.3f}%',end='\r')
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                if writer is not None:
                    writer.add_scalar('Loss/train_step', loss, epoch*trainsteps+trainingStep)
                
                preds = torch.argmax(outputs, dim=1)
                train_acc = accMetric(preds, labels)
                train_accs += train_acc

                if writer is not None:
                    writer.add_scalar('Accuracy/train_step', train_acc, epoch*trainsteps+trainingStep)
            if writer is not None:
                writer.add_scalar('Loss/train_epoch', train_loss/trainsteps, epoch)
                writer.add_scalar('Accuracy/train_epoch', train_accs/trainsteps, epoch)
            print(f'Training Loss: {train_loss/trainsteps:.4f} Training Accuracy: {train_accs/trainsteps:.4f}')
            self.validation_step(val_loader,criterion,device,accMetric,writer,epoch)