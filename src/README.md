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