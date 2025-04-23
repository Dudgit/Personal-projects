from dataclasses import dataclass
from torchvision import transforms
from PIL import Image

@dataclass
class Config:
  image_size = 224
  batch_size = 32
  num_epochs = 100
  learningRate = 1e-3 # Default ADAM rate
  trainSamples = 200 # Based on data exploration
config = Config()


exceptionList = [
  'C:\\Users\\bence\\.cache\\kagglehub\\datasets\\anshulmehtakaggl\\wildlife-animals-images\\versions\\43\\hyena-resize-224\\resize-224\\00000224_224resized.png']


class AnimalDataset:
  def __init__(self,allpath,labelMap,transfm = None) -> None:
    self.allpath = [path for path in allpath if path not in exceptionList]
    self.transform = transfm if transfm else transforms.Compose([transforms.Resize(config.image_size),transforms.ToTensor()])
    self.labelMap = labelMap
    self.reverseLabelMap = {v:k for k,v in labelMap.items()}

  def __len__(self)-> int:
    return len(self.allpath)

  def __getitem__(self,idx):
    sample_path = self.allpath[idx]
    img = Image.open(sample_path).convert('RGB')
    label = sample_path.split(f'-resize-{config.image_size}\\')[0].split('43\\')[-1]
    return self.transform(img), self.reverseLabelMap[label]
  
