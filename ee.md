# explore

## code structure

## run
- download dataset(1st) under input folder using kaggle cli;
- preprocess data; `python preprocess_dsb2018.py`
  - address the bug `AttributeError: 'NoneType' object has no attribute 'shape'`, by detect whether img is None.
- run;
```
time python train.py --dataset dsb2018_96 --arch NestedUNet
time python val.py --name dsb2018_96_NestedUNet_woDS
```

## code profiling
