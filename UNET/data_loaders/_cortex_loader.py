from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from ..datasets import *
from .base import *
import numpy as np

class CortexDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        self.data_dir = data_dir
        self.dataset = CortexDataset(self.data_dir)
        if not os.path.exists(self.data_dir + 'cortex'):
            self.dataset.fetch()
        else:
            self.dataset._read_csv()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)    
        
