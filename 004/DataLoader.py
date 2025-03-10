import os
import glob
import PIL.Image
import numpy as np
import torchvision
import tqdm
import PIL
from torch.utils.data import DataLoader

class PokeBall:
    def __init__(self, root:str = 'dataset',transform = None):
        self.root = root
        self.classes = os.listdir(root)
        self.class_to_idx = {self.classes[i]:i for i in range(len(self.classes))}
        self.idx_to_class = {i:self.classes[i] for i in range(len(self.classes))}

        self.allPath = glob.glob('dataset/*/*.png') +glob.glob('dataset/*/*.jpg') 
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) if not transform else transform

    def __len__(self):
        return len(self.allPath)
    
    def __transformImg__(self, img):
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        path = self.allPath[idx]
        path = path.replace('\\', '/')
        
        label = self.class_to_idx[path.split('/')[1]]
        img = PIL.Image.open(path).convert('RGB')
        return self.__transformImg__(img), label
    
    #def __get_batch__(self):
    #    imgs,lables = zip(*[self.__getitem__(idx)for idx in range(self.currentStep, self.currentStep + self.batch_size)])
    #    return np.array(imgs), lables        

    #def __next__(self):
    #    if self.currentStep + self.batch_size  < len(self):
    #        images, labels = self.__get_batch__()
    #        self.currentStep += self.batch_size
    #        return images, labels
    #    else:
    #        self.currentStep = 0
    #        np.random.shuffle(self.allPath)
    #        raise StopIteration
        
    def __iter__(self):
        return self
    
if __name__ == '__main__':
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.ToTensor()])
    pokeball = PokeBall(root = 'extra/dataset',batch_size = 32,transform=transforms)
    dataLoader = tor
    print(len(pokeball)//pokeball.batch_size)
    for batch in tqdm.tqdm(pokeball):
        continue#print(batch[0].shape,end='\r')