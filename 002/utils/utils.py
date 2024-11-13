from torchvision import transforms
from utils.initvars import img_targetSize

basetransforms = transforms.Compose([
    transforms.Resize(img_targetSize),
    transforms.ToTensor()
    # transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(20),
    #transforms.ToTensor()
])
traintransforms = transforms.Compose([
    transforms.Resize(img_targetSize),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(20),
    transforms.ToTensor()
])
