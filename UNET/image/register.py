import numpy as np
import matplotlib.pyplot as plt
import pims
from pystackreg import StackReg
from skimage.exposure import match_histograms
from skimage.io import imread, imsave
from skimage import transform, io, exposure
from glob import glob


def overlay_images(imgs, equalize=False, aggregator=np.mean):

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = np.stack(imgs, axis=0)

    return aggregator(imgs, axis=0)

def composite_images(imgs, equalize=False, aggregator=np.mean):

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = [img / img.max() for img in imgs]

    if len(imgs) < 3:
        imgs += [np.zeros(shape=imgs[0].shape)] * (3-len(imgs))

    imgs = np.dstack(imgs)

    return imgs

in_dir = '/home/cwseitz/Desktop/data/input/'
out_dir = '/home/cwseitz/Desktop/data/'
files = [
'/home/cwseitz/Desktop/data/input/20150618_wm9_A6_G3_NoDrug_hyb1-tile-0-0.tif',
'/home/cwseitz/Desktop/data/input/20150619_wm9_A6_G3_NoDrug_hyb2-tile-0-0.tif',
'/home/cwseitz/Desktop/data/input/20150621_wm9_A6_G3_NoDrug_hyb3-tile-0-0.tif',
'/home/cwseitz/Desktop/data/input/20150622_wm9_A6_G3_NoDrug_hyb4-tile-0-0.tif',
'/home/cwseitz/Desktop/data/input/20150623_wm9_A6_G3_NoDrug_hyb5-tile-0-0.tif'
]

dbbox = 400
unreg_full = np.zeros((5,9240,9240),dtype=np.float16); print(unreg_full.nbytes)
reg = np.zeros((5,2*dbbox,2*dbbox),dtype=np.float16)

"""
plt.imshow(imread(files[1])[0])
plt.show()
"""

x,y = (7740,7720)

unreg_full[0] = imread(files[0])[0]
unreg_full[1] = imread(files[1])[0]
unreg_full[2] = imread(files[2])[0]
unreg_full[3] = imread(files[3])[0]
unreg_full[4] = imread(files[4])[0]

unreg = unreg_full[:,x-dbbox:x+dbbox,y-dbbox:y+dbbox]
unreg[1] = match_histograms(unreg[1],unreg[0])
unreg[2] = match_histograms(unreg[2],unreg[0])
unreg[3] = match_histograms(unreg[3],unreg[0])
unreg[4] = match_histograms(unreg[4],unreg[0])
imsave(out_dir+'unreg-full.tif',unreg_full)

#Find transformation using a single cell(s) with ~constant axial position
reg[0] = unreg[0]
sr1 = StackReg(StackReg.TRANSLATION)
tmat1 = sr1.register(unreg[0], unreg[1])
reg[1] = sr1.transform(unreg[1],tmat=tmat1)

sr2 = StackReg(StackReg.TRANSLATION)
tmat2 = sr2.register(unreg[0], unreg[2])
reg[2] = sr2.transform(unreg[2],tmat=tmat2)

sr3 = StackReg(StackReg.TRANSLATION)
tmat3 = sr3.register(unreg[0], unreg[3])
reg[3] = sr3.transform(unreg[3],tmat=tmat3)

sr4 = StackReg(StackReg.TRANSLATION)
tmat4 = sr4.register(unreg[0], unreg[4])
reg[4] = sr4.transform(unreg[4],tmat=tmat4)
imsave(out_dir+'reg.tif',reg)
imsave(out_dir+'unreg.tif',unreg)


#Show whether or not the transformation works
fig, ax = plt.subplots(2,4)
ax[0,0].imshow(composite_images([unreg[0],unreg[1]]))
ax[0,1].imshow(composite_images([unreg[0],unreg[2]]))
ax[0,2].imshow(composite_images([unreg[0],unreg[3]]))
ax[0,3].imshow(composite_images([unreg[0],unreg[4]]))
ax[1,0].imshow(composite_images([reg[0],reg[1]]))
ax[1,1].imshow(composite_images([reg[0],reg[2]]))
ax[1,2].imshow(composite_images([reg[0],reg[3]]))
ax[1,3].imshow(composite_images([reg[0],reg[4]]))
plt.show()

#Apply transformation to entire image
sr = StackReg(StackReg.RIGID_BODY)
unreg_full[1] = sr.transform(unreg_full[1],tmat=tmat1)
unreg_full[2] = sr.transform(unreg_full[2],tmat=tmat2)
unreg_full[3] = sr.transform(unreg_full[3],tmat=tmat3)
unreg_full[4] = sr.transform(unreg_full[4],tmat=tmat4)

#Output
imsave(out_dir+'reg-full.tif',unreg_full)


	
