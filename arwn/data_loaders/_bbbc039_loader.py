from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from ..datasets import *
from .base import *
import numpy as np

class U2OSDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, crop_dim=256, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        transform = transforms.Compose([
           transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
           transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = U2OSDataset(self.data_dir, crop_dim, transform=transform, target_transform=target_transform)
        if not os.path.exists(self.data_dir + 'bbbc039'):
            self.dataset.fetch()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
  
class U2OSBoundaryDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, crop_dim=256, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        transform = transforms.Compose([
           transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
           transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = U2OSBoundaryDataset(self.data_dir, crop_dim, transform=transform, target_transform=target_transform)
        if not os.path.exists(self.data_dir + 'bbbc039'):
            self.dataset.fetch()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)      
        
