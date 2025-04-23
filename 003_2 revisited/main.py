import kagglehub
path = kagglehub.dataset_download("anshulmehtakaggl/wildlife-animals-images")

from glob import glob
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader
from utils import config, AnimalDataset

import torch
from zoo import AnimalModel, Trainer
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

label_map = {0:'cheetah',1:'fox',2:'hyena',3:'lion',4:'tiger',5:'wolf'}

def clearPath(allPaths):
  for ap in allPaths:
    for path in ap:
        try:
            Image.open(path)
        except:
            print(f"Corrupted image: {path}")
            ap.remove(path)
            continue
    return allPaths

def getPaths():
    classPaths = glob(f'{path}/*-{config.image_size}')
    allPaths = []
    for cp in classPaths:
        allPaths.append(glob(f'{cp}/*{config.image_size}/*'))
    allPaths = clearPath(allPaths)
    trainPaths =  np.array([ap[:config.trainSamples] for ap in allPaths]).flatten()
    valPaths =  np.array([ap[config.trainSamples:config.trainSamples+50] for ap in allPaths]).flatten()
    return trainPaths, valPaths

def getLoader(trainPaths,valPaths):
    trainDataset = AnimalDataset(trainPaths,labelMap=label_map)
    trainLoader = DataLoader(trainDataset,batch_size=config.batch_size,shuffle=True)

    valDataset = AnimalDataset(valPaths,labelMap=label_map)
    valLoader = DataLoader(valDataset,batch_size=config.batch_size,shuffle=False)
    return trainLoader,valLoader



def main():
    trainPaths,valPaths = getPaths()
    trainLoader,valLoader = getLoader(trainPaths,valPaths)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AnimalModel(num_classes=len(label_map)).to(device)
    criterion = F.cross_entropy
    model = model.to(device)
    writer = SummaryWriter()
    trainer = Trainer(criterion=criterion,device = device,writer = writer)
    trainer.train(model,train_loader=trainLoader,num_epochs=config.num_epochs,val_loader=valLoader)
    writer.close()

if __name__ == "__main__":
    main()