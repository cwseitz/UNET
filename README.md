# UNET

The library implements the UNET architecture in PyTorch for semantic segmentation of microscopy images. Like most PyTorch projects, it is implemented as a object-oriented framework. As such, you will need to write new objects to tailor it to your specific needs. In PyTorch data is a ```Dataset``` object, the object that loads the data is a ```DataLoader```, the model itself (UNET) is a ```Model```, the trainer the trains the model is a ```Trainer```, etc.


## Basic installation

All of the necessary dependencies are specified in the conda environment file unet.yml. Assuming you have conda already installed on your machine, run
 
``` 
conda env create -f /path/to/unet.yml
conda activate unet
```  

To make sure everything is working correctly, you can navigate to UNET/UNET/examples. There you will find an example named 'bbbc039.py' which segments a large dataset of pre-annotated U2OS nuclei procured by the BROAD institute. Run the training process on this example to ensure everything is working correctly:

``` 
python bbbc039.py
```  

The example illustrates the major components of the framework. I'll walk through the code piece by piece. The first block builds a configuration object ```config``` from parameters in the file 'bbbc039.json'. 

``` 
config_path = 'bbbc039.json'
file = open(config_path)
config = json.load(file)
config = ConfigParser(config)
```  

For every new project you need a configuration file like this one. Next we prepare the device we are going to the model on:

``` 
n_gpu = 0
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

If you want to learn more about what these are, Finally, instantiate the object ```UNETTrainer``` and run the training process

``` 
trainer = UNETTrainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
                      
                      
trainer.train()
```

