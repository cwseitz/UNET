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

class U2OSDataset(Dataset):
	def __init__(self, dir, crop_dim, transform=None, target_transform=None):
		self.bbbc_dir = dir + 'bbbc039'
		self.img_dir = self.bbbc_dir  + '/images'
		self.tgt_dir = self.bbbc_dir  + '/masks'
		self.met_dir = self.bbbc_dir + '/meta'
		self.transform = transform
		self.target_transform = target_transform
		self.crop_dim = crop_dim

	def __len__(self):
		return len(os.listdir(self.img_dir))

	def __getitem__(self, idx):
		files = os.listdir(self.img_dir)
		img_path = os.path.join(self.img_dir, files[idx])
		tgt_path = self.swap_ext(os.path.join(self.tgt_dir, files[idx]))
		image = imread(img_path).astype(np.float64)
		image = (image-image.mean())/image.std()
		targt = self.split_mask(imread(tgt_path))
		
		#random crop
		start_dim1 = np.random.randint(low=0, high=image.shape[0] - self.crop_dim)
		start_dim2 = np.random.randint(low=0, high=image.shape[1] - self.crop_dim)
		image = image[start_dim1:start_dim1 + self.crop_dim, start_dim2:start_dim2 + self.crop_dim]
		targt = targt[start_dim1:start_dim1 + self.crop_dim, start_dim2:start_dim2 + self.crop_dim]
		
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			targt = self.target_transform(targt)
		return image, targt

	def fetch(self):
		os.makedirs(self.bbbc_dir, exist_ok=True)
		ext_imgs_url = 'https://data.broadinstitute.org/bbbc/BBBC039/images.zip'
		ext_mask_url = 'https://data.broadinstitute.org/bbbc/BBBC039/masks.zip'
		ext_meta_url = 'https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip'
		self.request_and_extract(ext_imgs_url, self.bbbc_dir, name='images')
		self.request_and_extract(ext_mask_url, self.bbbc_dir, name='masks')
		self.request_and_extract(ext_meta_url, self.bbbc_dir, name='meta')

	def request_and_extract(self, url, dir, name='images'):
		zip = requests.get(url)
		path_to_zip = dir + '/%s.zip' % (name)
		#save zip
		with open(path_to_zip, 'wb') as f:
			f.write(zip.content)
		#extract zip
		with ZipFile(path_to_zip, 'r') as zip_obj:
			zip_obj.extractall(path=dir)

	def swap_ext(self, path):
		return '.'.join([path.split('.')[0],'png'])

	def split_mask(self, mask, min_size=100, boundary_size=2):
		mask = rgb2gray(mask[:,:,:3]) #remove alpha channel
		boundaries = find_boundaries(mask)

		proc_mask = np.zeros((mask.shape + (3,)))
		proc_mask[(mask == 0) & (boundaries == 0), 0] = 1
		proc_mask[(mask != 0) & (boundaries == 0), 1] = 1
		proc_mask[boundaries == 1, 2] = 1

		return proc_mask

class U2OSBoundaryDataset(Dataset):
	def __init__(self, dir, crop_dim, transform=None, target_transform=None):
		self.bbbc_dir = dir + 'bbbc039'
		self.img_dir = self.bbbc_dir  + '/images'
		self.tgt_dir = self.bbbc_dir  + '/masks'
		self.met_dir = self.bbbc_dir + '/meta'
		self.transform = transform
		self.target_transform = target_transform
		self.crop_dim = crop_dim

	def __len__(self):
		return len(os.listdir(self.img_dir))

	def __getitem__(self, idx):
		files = os.listdir(self.img_dir)
		img_path = os.path.join(self.img_dir, files[idx])
		tgt_path = self.swap_ext(os.path.join(self.tgt_dir, files[idx]))
		image = imread(img_path).astype(np.float64)
		image = (image-image.mean())/image.std()
		targt = self.split_mask(imread(tgt_path))
		
		#random crop
		start_dim1 = np.random.randint(low=0, high=image.shape[0] - self.crop_dim)
		start_dim2 = np.random.randint(low=0, high=image.shape[1] - self.crop_dim)
		image = image[start_dim1:start_dim1 + self.crop_dim, start_dim2:start_dim2 + self.crop_dim]
		targt = targt[start_dim1:start_dim1 + self.crop_dim, start_dim2:start_dim2 + self.crop_dim]
		
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			targt = self.target_transform(targt)
		return image, targt

	def fetch(self):
		os.makedirs(self.bbbc_dir, exist_ok=True)
		ext_imgs_url = 'https://data.broadinstitute.org/bbbc/BBBC039/images.zip'
		ext_mask_url = 'https://data.broadinstitute.org/bbbc/BBBC039/masks.zip'
		ext_meta_url = 'https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip'
		self.request_and_extract(ext_imgs_url, self.bbbc_dir, name='images')
		self.request_and_extract(ext_mask_url, self.bbbc_dir, name='masks')
		self.request_and_extract(ext_meta_url, self.bbbc_dir, name='meta')

	def request_and_extract(self, url, dir, name='images'):
		zip = requests.get(url)
		path_to_zip = dir + '/%s.zip' % (name)
		#save zip
		with open(path_to_zip, 'wb') as f:
			f.write(zip.content)
		#extract zip
		with ZipFile(path_to_zip, 'r') as zip_obj:
			zip_obj.extractall(path=dir)

	def swap_ext(self, path):
		return '.'.join([path.split('.')[0],'png'])

	def split_mask(self, mask, boundary_size=2):
		mask = rgb2gray(mask[:,:,:3]) #remove alpha channel
		boundaries = find_boundaries(mask)

		proc_mask = np.zeros_like(mask)
		proc_mask[boundaries == 1] = 1

		return proc_mask
