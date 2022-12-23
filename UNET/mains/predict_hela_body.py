import argparse
import collections
import torch
import numpy as np
import tifffile
import torch.nn.functional as F
import matplotlib.pyplot as plt
import UNET.torch_models as module_arch
import UNET.data_loaders as data_loaders
from UNET.utils import ConfigParser
from UNET.utils import prepare_device
from UNET.torch_models import UNetModel
from torchsummary import summary
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.util import img_as_int
from skimage.restoration import rolling_ball
torch.cuda.empty_cache()

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def plot(output,tiff):
    output = F.softmax(output,dim=1)
    idx = output.argmax(dim=1)
    mask = F.one_hot(idx,num_classes=3)
    mask = torch.permute(mask,(0,3,1,2))
    nmask = mask[0,1,:,:].numpy()
    nmask = remove_small_holes(nmask,area_threshold=64)
    nmask = label(nmask)
    nmask = clear_border(nmask)
    nmask = remove_small_objects(nmask,min_size=200)
    nmask[nmask > 0] = 1
    nmask = label(nmask)
    fig, ax = plt.subplots(1,2,sharex=True,sharey=True)
    ax[0].imshow(nmask,cmap='gray')
    ax[1].imshow(tiff,cmap='gray')
    plt.show()

def resize_and_blend(ch0,ch1,new_size=(256,256)):
    ch0 = resize(ch0,new_size)
    ch1 = resize(ch1,new_size)
    r = ch0.max()/ch1.max()
    ch1 *= r
    blended = 0.4*ch0 + 0.6*ch1
    blended = gaussian(blended,sigma=1)
    blended = rolling_ball(blended,radius=50)
    blended = img_as_int(2*blended)
    return blended

def main(config,path,ch0_path,ch1_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.init_obj('arch', module_arch)
    model.to(device=device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    ch0 = tifffile.imread(ch0_path).astype(np.int16)
    ch1 = tifffile.imread(ch1_path).astype(np.int16)
    nt,nx,ny = ch0.shape
    for n in range(nt):
        with torch.no_grad():
            tiff = resize_and_blend(ch0[n],ch1[n])
            image = torch.from_numpy(tiff).unsqueeze(0).unsqueeze(0)
            image = image.to(device=device, dtype=torch.float)
            output = model(image).cpu()
            plot(output,tiff)
            torch.cuda.empty_cache()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    stack_path = '/research3/shared/cwseitz/Analysis/221218-Hela-IFNG-16h-2_1/'
    ch0_file = '221218-Hela-IFNG-16h-2_1_mxtiled_corrected_stack_ch0.tif'
    ch1_file = '221218-Hela-IFNG-16h-2_1_mxtiled_corrected_stack_ch1.tif'
    path = 'saved/models/UNetModel/1222_162246/model_best.pth'
    main(config,path,stack_path+ch0_file,stack_path+ch1_file)
