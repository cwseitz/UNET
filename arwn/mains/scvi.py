import argparse
import collections
import torch
import numpy as np
import matplotlib.pyplot as plt

import arwn.data_loaders as data_loaders
import arwn.torch_models as module_arch
import arwn.train.loss as module_loss

from arwn.utils import ConfigParser
from arwn.utils import prepare_device
from arwn.train import SCVITrainer
from torchsummary import summary
torch.cuda.empty_cache()

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):

    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', data_loaders)
    valid_data_loader = data_loader.split_validation()
    nsamples, nvars = data_loader.dataset.data.shape

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    #torch.cuda.set_per_process_memory_fraction(0.5, device=device)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
       
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = SCVITrainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


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
    main(config)
