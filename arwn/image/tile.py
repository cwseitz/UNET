import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from glob import glob


def cm_stitch(M,N,L,K,im_rows,im_cols,in_dir,out_dir,channels,ovrlp,expmt):

	"""
	Assumes the image files are indexed in column-major order and are not snaked
	"""
	
	for m in range(M):	
		for n in range(N):

			row_offset, col_offset = (m*L,n*L)
			blk_shape = (len(channels),L*im_rows-ovrlp*L,K*im_cols-ovrlp*K)
			print(f'Building block: ({m},{n}) with shape {blk_shape}')
			blk = np.zeros(blk_shape,dtype=np.float16)

			for l in range(L):
				for k in range(K):
					print(f'Building tile: ({l},{k}) with shape {im_rows, im_cols}')
					x,y = (row_offset+l,col_offset+k)
					idx = x + y*nrows
					fidx = idx + 1 #file index
					idx_fmt = "{0:0=3d}".format(fidx)

					r = (l*(im_rows-ovrlp),(l+1)*(im_rows-ovrlp))
					s = (k*(im_cols-ovrlp),(k+1)*(im_cols-ovrlp))

					for c in range(len(channels)):
						channel = channels[c]
						#cutting out top and left to keep only first pass
						blk[c,r[0]:r[1],s[0]:s[1]] =\
						imread(in_dir + channel + idx_fmt + '.TIF')[ovrlp:,ovrlp:]

			imsave(out_dir + f'{expmt}-tile-{m}-{n}.tif',blk)
			del blk

expmt = '20150623_wm9_A6_G3_NoDrug_hyb5'
in_dir = '/media/cwseitz/Data/CancerData Dropbox/Clayton Seitz/' + expmt + '/'
out_dir = '/home/cwseitz/Desktop/'
channels = ['dapi', 'alexa', 'cy', 'tmr', 'nir','nir_sac']

nrows = 44
ncols = 43
ovrlp = 100 #pixels
tile_dim = (nrows,ncols)
im_rows, im_cols = (1024,1024)
L = K = 10 #tiles must be square
M = N = min(nrows,ncols) % L

cm_stitch(M,N,L,K,im_rows,im_cols,in_dir,out_dir,channels,ovrlp,expmt)







	
