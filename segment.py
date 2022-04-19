import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import watershed
from skimage.color import gray2rgb, rgb2gray, label2rgb
from skimage.filters import threshold_niblack, threshold_otsu, sobel
from skimage.util import img_as_ubyte
from skimage.io import imread
from scipy import ndimage as ndi
from glob import glob

def segment_otsu(image):
	"""Segmentation using the Otsu threshold"""
	threshold = threshold_otsu(image)
	mask = image > threshold
	return mask

def niblack(image,window_size=25):

	thresh = threshold_niblack(image, window_size=window_size, k=0)
	return thresh

def wtshd(image,markers,plot=True):

	"""Segmentation using the watershed method"""

	mark_arr = np.zeros_like(image,dtype=bool)
	mark_arr[markers[:,1],markers[:,0]] = True
	mark_arr = label(mark_arr)
	binary = segment_otsu(image)
	distance = ndi.distance_transform_edt(binary)
	mask = watershed(-distance,mark_arr,mask=binary)

	if plot:

		fig, axes = plt.subplots(1,4,figsize=(9,3), sharex=True, sharey=True)
		ax = axes.ravel()

		ax[0].imshow(image, cmap=plt.cm.gray)
		ax[1].imshow(binary, cmap=plt.cm.gray)
		ax[2].imshow(distance, cmap=plt.cm.gray)
		ax[3].imshow(mask,cmap='gray')

		for a in ax:
			a.set_axis_off()

		fig.tight_layout()
		plt.show()

	return mask

def get_props(label_mask,image,props=('bbox','label','area','perimeter','centroid')):
	"""Get the default region properties as a DataFrame"""

	table = regionprops_table(label_mask,image,properties=props)
	df = pd.DataFrame(table)
	return df

def add_circ(df):
	df['circ'] = 4*np.pi*df['area']/df['perimeter']**2
	return df

def onclick(event):
	global ix, iy
	ix, iy = event.xdata, event.ydata

	global coords
	if ix != None and iy != None:
		ix = int(round(ix)); iy = int(round(iy))
		coords.append([ix, iy])

def view_region(image,labeled_mask,df,label,delta=50):

	"""Iterate over regions and view within the bbox"""

	row = df.loc[df['label'] == label]
	min_row, min_col, max_row, max_col =\
	row[['bbox-0','bbox-1','bbox-2','bbox-3']].to_numpy()[0].astype(np.int32)
	cx, cy = row[['centroid-0','centroid-1']].to_numpy()[0].astype(np.int32)
	#patch = image[min_row:max_row,min_col:max_col]
	#patch_lbl = labeled_mask[min_row:max_row,min_col:max_col]
	patch = cut_square_patch(image,min_row,max_row,min_col,max_col)
	patch_lbl = cut_square_patch(labeled_mask,min_row,max_row,min_col,max_col)

	overlay = label2rgb(patch_lbl, image=patch, bg_label=0,alpha=0.05)
	overlay /= overlay.max()

	fig, ax = plt.subplots(1,3,figsize=(10,4))
	circ, avg_lap = row['circ'], row['avg_lap']
	patcht = np.abs(fftshift(fft2(patch)))

	circular_mask = np.ones_like(patcht)
	rad = int(patcht.shape[0]/4)
	xx, yy = int(patcht.shape[0]/2), int(patcht.shape[1]/2)
	rr, cc = ellipse(xx,yy,rad,rad)
	circular_mask[rr,cc] = 0
	idx = np.where(circular_mask > 0)
	avg = np.mean(patcht[idx])

	props = dict(boxstyle='round', facecolor='white', alpha=0.5)
	textstr = '\n'.join((
		r'$circ=%.6f$' % (circ, ),
		r'$lap=%.6f$' % (avg, )))

	ax[0].imshow(overlay)
	ax[0].text(0.45, 0.1, textstr, transform=ax[0].transAxes, fontsize=14,
			 bbox=props)
	d = compute_laplace(gaussian(patch))
	ax[1].imshow(circular_mask)
	ax[2].imshow(patcht*circular_mask)
	plt.tight_layout()
	plt.show()

def view_all_regions(im,mask,df):
	"""Iterate over labels and call view_regions()"""
	labels = df['label'].unique()
	for label in labels:
		view_region(im,mask,df,label)

def mark_nuclei(image,coords):

	sub = niblack(image)
	fig, ax = plt.subplots()
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	ax.imshow(sub,cmap='gray')
	plt.show()

	coords = np.array(coords)
	fig, ax = plt.subplots()
	ax.imshow(sub,cmap='gray')
	ax.scatter(coords[:,0],coords[:,1],color='red',s=3)
	plt.show()

	return coords

dir = '/media/cwseitz/Data/CancerData Dropbox/Clayton Seitz/20150618_wm9_A6_G3_NoDrug_hyb1/'
files = glob(dir + '*dapi*')
for file in files:
	im = imread(file)
	coords = []
	coords = mark_nuclei(im,coords)
	wtshd(im, coords)
	#df = get_props(labeled_mask,im)
	#df = add_laplace(df,im,labeled_mask)
	#df = add_circ(df)
	#view_all_regions(im,labeled_mask,df)
	#del im, labeled_mask

