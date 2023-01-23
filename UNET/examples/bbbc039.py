import argparse
import collections
import torch
import json
import numpy as np
import UNET.data_loaders as data_loaders
import UNET.train.loss as module_loss
import UNET.train.metrics as module_metric
import matplotlib.pyplot as plt
import UNET.torch_models as module_arch
from UNET.utils import ConfigParser
from UNET.utils import prepare_device
from UNET.train import UNETTrainer

config_path = 'config.json'
file = open(config_path)
config = json.load(file)
config = ConfigParser(config)

logger = config.get_logger('train')
data_loader = config.init_obj('data_loader', data_loaders)
valid_data_loader = data_loader.split_validation()
model = config.init_obj('arch', module_arch)
logger.info(model)

n_gpu = 0
device, device_ids = prepare_device(n_gpu)
model = model.to(device)
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

criterion = getattr(module_loss, config['loss'])
metrics = [getattr(module_metric, met) for met in config['metrics']]
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

trainer = UNETTrainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
                      
                      
trainer.train()
