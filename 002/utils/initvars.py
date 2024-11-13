import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classidxs = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
batch_size = 64
rootf = 'data/seg_'
img_targetSize = (128, 128)