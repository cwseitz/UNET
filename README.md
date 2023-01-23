# UNET

The library implements the UNET architecture in PyTorch for semantic segmentation of microscopy images. Like most PyTorch projects, it is implemented as a object-oriented framework. As such, you will need to write new objects to tailor it to your specific needs. In PyTorch data is a ```Dataset``` object, the object that loads the data is a ```DataLoader```, the model itself (UNET) is a ```Model```, the trainer the trains the model X is a ```XTrainer```, etc.


## Basic installation

All of the necessary dependencies are specified in the conda environment file unet.yml. Assuming you have conda already installed on your machine, run
 
``` 
conda env create -f /path/to/unet.yml
conda activate unet
```  

To make sure everything is working correctly, you can navigate to ```UNET/UNET/examples```. There you will find an example named 'bbbc039.py' which trains a large dataset of pre-annotated U2OS nuclei procured by the BROAD institute. 

The example illustrates the major components of the framework, so I'll walk through the code piece by piece. The first block builds a configuration object ```config``` from parameters in the file 'bbbc039.json'. 


Let's take a quick look at the configuration file 'bbbc039.json'

```
{
		"name": "UNetModel",
		"n_gpu": 0,

		"arch": {
				"type": "UNetModel",
				"args": {
		"n_channels": 1,
		"n_classes": 3
	}
		},
		"data_loader": {
				"type": "U2OSDataLoader",
				"args":{
						"data_dir": "data/",
						"batch_size": 5,
						"shuffle": true,
						"validation_split": 0.2,
						"num_workers": 2,
						"crop_dim": 256
				}
		},
		"optimizer": {
				"type": "Adam",
				"args":{
						"lr": 0.01,
						"weight_decay": 0,
						"amsgrad": true
				}
		},
		"loss": "CrossEntropyLoss",
		"metrics": ["BackgroundRecall",
								"BoundaryRecall",
								"InteriorRecall",
								"BackgroundPrecision",
								"BoundaryPrecision",
								"InteriorPrecision"],
		"lr_scheduler": {
				"type": "StepLR",
				"args": {
						"step_size": 1,
						"gamma": 1
				}
		},
		"trainer": {
				"epochs": 50,

				"save_dir": "saved/",
				"save_period": 1,
				"verbosity": 2,

				"monitor": "min val_loss",
				"early_stop": 10,
				"tensorboard": true
		}
}

```

As you can see, we set ```arch``` to ```UNetModel``` which is because we want to use the model architecture specified by ```UNET.torch_models.UNetModel```. The remaining json keys e.g., ```data_loader```, contain the other objects we need for training, and their associated parameterization. Note that we are using ```U2OSDataLoader``` because we want to use the BBBC039 dataset. 


Like everything else, your configuration itself is an object. For every new application, you should make a configuration file like this one. In the code, we build the configuration using 

``` 
config_path = 'bbbc039.json'
file = open(config_path)
config = json.load(file)
config = ConfigParser(config)
```  

Then we build a logger, data loader, and the model itself:

logger = config.get_logger('train')
data_loader = config.init_obj('data_loader', data_loaders)
valid_data_loader = data_loader.split_validation()
model = config.init_obj('arch', module_arch)
logger.info(model)

Next we prepare the device we are going to the model on:

``` 
device, device_ids = prepare_device(n_gpu)
model = model.to(device)
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)
```

We set ```n_gpu = 0``` in this case because we will train on the CPU. If you have GPU available set ```n_gpu = 1```. Next we set up the criteria we will train on 

``` 
criterion = getattr(module_loss, config['loss'])
metrics = [getattr(module_metric, met) for met in config['metrics']]
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
```

AS you can see, these are also specified in the configuration file. They are required, but you can try different optimizers, learning rate schedulers etc  if desired. Finally, instantiate the object ```UNETTrainer``` and run the training process

``` 
trainer = UNETTrainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
                      
                      
trainer.train()
```

