# Source code
This the early version of source code for developing our inpainting model.

Some of the files are too large to upload. We strongly recommend to download the src from [google drive](https://drive.google.com/file/d/1mh5t17vaR1GcL44iMyavqRDMbTmo665Z/view?usp=sharing)
or [baidu](https://pan.baidu.com/s/1eXc2elmsY2t__mJRKI_l2g) with password: w5r4.



# Enviroment
tensorflow 1.12 and other dependencies (install if needed)

# Config file
inpaint_config.yml

# Data
Prepare *xxx.flist* in **/data** using **flist/flist.py**

# Mask
1. download from https://drive.google.com/file/d/140bV9FlOnnBbG4L4OiiqMmAbOQ09bQH7/view?usp=sharing
2. prepare *xxx.flist* in **/data**;
3. partially generate from scratch, defined in *utils_fn.py*

# Train in terminal
**IMPORTANT:** config *inpaint_config.yml* correctly
```
python train_inpaint_model.py
```

# Validation or test in terminal
1. set TEST_NUM in *inpaint_config.yml*
2. set MODEL_RESTORE in *inpaint_config.yml*
```
python val_inpaint_model.py
```

# Tensorboard
```
python -m tensorard.main --logdir=./logs
```

# Evaluation final metrics 
After run ```python val_inpaint_model.py```, then run
```
python evaluation/evaluation.py 
```
