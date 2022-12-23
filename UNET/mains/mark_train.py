import matplotlib.pyplot as plt
import numpy as np
import napari 
import tifffile
from skimage.io import imsave
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.util import img_as_int
from skimage.restoration import rolling_ball
from skimage.filters import threshold_otsu

class TrainDataGenerator:
    def __init__(self,dir,file):
        self.dir = dir
        self.file = file
        self.stack = tifffile.imread(dir+file)
        self.pfx = file.split('.')[0]
    def mark_boundaries(self):
        nt,nx,ny = self.stack.shape
        for n in range(nt):
            image = self.stack[n]
            viewer = napari.Viewer()
            viewer.window.resize(2000, 1000)
            thresh = threshold_otsu(image)
            mask0 = np.zeros_like(image)
            mask0[image > thresh] = 1
            viewer.add_image(image,name='Membrane',colormap='green')
            viewer.add_shapes(name='Mask')
            napari.run()
            this_shape = image.shape
            labels = viewer.layers['Mask'].to_labels(labels_shape=this_shape).astype(np.uint16)
            imsave(self.dir + self.pfx + f'_mask_{n}.tif',labels)


dir = '/research3/shared/cwseitz/Analysis/221218-Hela-IFNG-16h-2_1/' 
file = '221218-Hela-IFNG-16h-2_1_mxtiled_corrected_stack_blended.tif'
train_generator = TrainDataGenerator(dir,file)
train_generator.mark_boundaries()
