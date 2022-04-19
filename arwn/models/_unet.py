import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn.functional as F
import matplotlib.pyplot as plt

from skimage.measure import label
from skimage.io import imread
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import find_boundaries
from skimage.color import rgb2gray
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class UNetModel(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = 3
        self.padding = 1
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.double_conv.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal(m.weight, mean=0.0, std=np.sqrt(2/(self.in_channels*self.kernel_size**2)))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.kernel_size = 2
        self.in_channels = in_channels
        self.stride = 2
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=self.kernel_size, stride=self.stride)
            torch.nn.init.normal(self.up.weight, mean=0.0, std=np.sqrt(2/(in_channels*self.kernel_size**2)))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.kernel_size = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size)
        torch.nn.init.normal(self.conv.weight, mean=0.0, std=np.sqrt(2/(in_channels*self.kernel_size**2)))
    def forward(self, x):
        return self.conv(x)
        
        

def unet_weight_map(y, wc=None, w0=10, sigma=5):

    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.
    
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf
    
    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.
    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """
    
    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
        
        if wc:
            class_weights = np.zeros_like(y)
            for k, v in wc.items():
                class_weights[y == k] = v
            w = w + class_weights
    else:
        w = np.zeros_like(y)
    
    return w

def CrossEntropyLoss(output, target, diagnostic=True):

    if diagnostic: 
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(output[0,0,:,:].detach().cpu().numpy())
        ax[1].imshow(target[0,0,:,:].detach().cpu().numpy())
        plt.show()
    return F.cross_entropy(output,target)

def WeightedCrossEntropyLoss(output, target, weighted=False, diagnostic=False):
    """
    Reduces output to a single loss value by averaging losses at each element
    by default in (PyTorch 1.11.0)
    
    ***Note: this function operates on non-normalized probabilities
    (there is no need for a softmax function, it is included in the loss).
    
    At testing time or when computing metrics, you will need to implement a softmax layer.
    """
    
    wc = {
    0: 0, # background
    1: 0  # objects
    }
    
    mat = target.cpu().numpy()[:,1,:,:]
    
    if weighted:
        weight = torch.zeros_like(target[:,1,:,:])
        for idx in range(mat.shape[0]):
    	    weight[idx] = torch.from_numpy(unet_weight_map(mat[idx], wc))
    else:
        weight = torch.ones_like(target[:,1,:,:])

    logp = -torch.log(torch.sum(torch.mul(F.softmax(output,dim=1),target),dim=1))
    if torch.cuda.is_available():
    	weight = weight.cuda()

    loss_map = torch.mul(logp,weight)

    if diagnostic:
        im1 = torch.mul(F.softmax(output,dim=1),target)[0].permute(1,2,0).cpu().detach().numpy()
        im2 = target[0].permute(1,2,0).cpu().detach().numpy()
        im3 = loss_map[0].cpu().detach().numpy()
        fig, ax = plt.subplots(1,3,sharex=True,sharey=True)
        ax[0].imshow(im2,cmap='gray')
        ax[0].set_title('target')
        ax[1].imshow(im1,cmap='gray')
        ax[1].set_title('output')
        ax[2].imshow(10*im3,cmap='coolwarm')
        ax[2].set_title('weights')
        for x in ax:
            x.axis('off')
        plt.show()
    loss = torch.mean(loss_map)

    return loss
    
class MetricTools(object):
	def __init__(self):
		pass
	@staticmethod
	def _get_class_data(gt, pred, class_idx):
		#class_pred = pred[:, class_idx, :, :]
		#class_gt = gt[:, class_idx, :, :]
		class_pred = pred; class_gt = gt
		pred_flat = class_pred.contiguous().view(-1, )
		gt_flat = class_gt.contiguous().view(-1, )
		tp = torch.sum(gt_flat * pred_flat)
		fp = torch.sum(pred_flat) - tp
		fn = torch.sum(gt_flat) - tp

		tup = tp.item(), fp.item(), fn.item()

		return tup

def BackgroundPrecision(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 0)
	precision = tp/(tp+fp+eps)

	return precision

def InteriorPrecision(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 1)
	precision = tp/(tp+fp+eps)

	return precision

def BoundaryPrecision(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 2)
	recall = tp/(tp+fp+eps)

	return recall

def BackgroundRecall(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 0)
	recall = tp/(tp+fn+eps)

	return recall

def InteriorRecall(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 1)
	recall = tp/(tp+fn+eps)

	return recall

def BoundaryRecall(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 2)
	recall = tp/(tp+fn+eps)

	return recall
