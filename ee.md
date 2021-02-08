# explore
## time
- setup and run code, 1h30min, Feb 5th
- the whole code review and profiling --> mastery, 2h30min, Feb 8th.

## code structure
check code profiling.

## run
- download dataset(1st) under input folder using kaggle cli;
- preprocess data; `python preprocess_dsb2018.py`
  - address the bug `AttributeError: 'NoneType' object has no attribute 'shape'`, by detect whether img is None.
- run; Note: on hpc3, run w. sbatch.
```
time python train.py --dataset dsb2018_96 --arch NestedUNet
time python val.py --name dsb2018_96_NestedUNet_woDS
```

## code profiling 

- train.py and val.py, main files; focus on main.py
  - clear structure supporting many training choices: model, loss, dataset, optimizer, scheduler settings.
  - basic code logic
    - config load and handling
    - create loss
    - create model
    - create optimizer, scheduler
    - load dataset w. tranformations to obtain data loader
    - training(gradient descent) to learn parameters by backprop w.r.t model parameters.
    - log
    - serialize w. best iou, and support early stopping.

- archs.py; elegant code implementation.
  - VGGBlock, 2 consecutive conv w. BN.
  - UNet, a symmetrical network, composed of conv, upsampling, concat and 1x1 conv operations.
  - NestedUNet, Unet improved version w. dense connection and deep supervision; (**basic logic is implement Unet+L{1,2,3,4}, then conduct dense connections for each final output**)

- losses, metrics.py
- dataset.py and preprocess_dsb2018.py


## what can be used?
- how to define model/metric/loss classes in a py file w. `__all__` list
- use advanced optimizer;
  - Adam(params, lr, weight_decay)
  - SGD(params,lr,momentum,nesterov,weight_decay)
- use advanced scheduler;
  - No scheudler;
  - CosineAnnealingLR;
  - ReduceLROnPlateau(optimizer, factor, patience, min_lr)
  - MultiStepLR(optimizer, milestones, gamma), split epochs into 3 ranges, 1st range use init_lr, 2nd use init_lr*factor, ...
- early stopping; logic is sim. to TF2
- use OrderedDict to store logs and serilize w. pandas
  - OrderedDict diff. w. Dict on considering key insertion sequence.
- tqdm use to monitor the progress of the program
  - pbar=tqdm(total=100)
  - pbar.set_postfix(ordered_dict)
  - pbar.update(1)
  - pabar.close()

