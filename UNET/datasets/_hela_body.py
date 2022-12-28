import os
import numpy as np
import requests
import random
import warnings
from os.path import join
from glob import glob
from zipfile import ZipFile
from numpy.random import randint
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.segmentation import find_boundaries
from skimage.morphology import remove_small_objects

class HelaBodyDataset(Dataset):
	def __init__(self, dir, crop_dim, transform=None, target_transform=None):
		self.img_dir = dir  + '/images'
		self.tgt_dir = dir  + '/masks'
		self.transform = transform
		self.target_transform = target_transform
		self.crop_dim = crop_dim

	def __len__(self):
		return len(os.listdir(self.img_dir))

	def __getitem__(self, idx):
		files = sorted(os.listdir(self.img_dir))
		img_path = os.path.join(self.img_dir, files[idx])
		tgt_path = self.swap_suffix(os.path.join(self.tgt_dir, files[idx]))
		image = imread(img_path).astype(np.float64)
		image = (image-image.mean())/image.std()
		target = self.split_mask(imread(tgt_path))
				
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)
		return image, target

	def swap_suffix(self, path):
		return path.replace('_blended_','_blended_mask_')

	def split_mask(self, mask, min_size=100, boundary_size=2):
		boundaries = find_boundaries(mask)
		proc_mask = np.zeros((mask.shape + (3,)))
		proc_mask[(mask == 0) & (boundaries == 0), 0] = 1
		proc_mask[(mask != 0) & (boundaries == 0), 1] = 1
		proc_mask[boundaries == 1, 2] = 1
		return proc_mask
		
		
