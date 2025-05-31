## Requirements
* Python 3.7.4
* PyTorch 1.7.0
* CUDA 10.2


## Dataset
The expected structure of files is:
```
|-- First_stage
     |-- checkpoint
     |-- data
     |-- fewshot_re_kit
     |-- logs
     |-- models
     |-- train_demo.py
     |-- run_bert.sh
 
|-- Second_stage
     |-- checkpoint
     |-- data
     |-- fewshot_re_kit
     |-- logs
     |-- models
     |-- train_demo.py
     |-- run_bert.sh
```

## Train
At the first stage, you can train a encoder by:
```
sh ./First_stage/run_bert.sh
```
In the file ``run_bert.sh``, we can modify the settings and hyper-parameters of the model.


At the second stage, you can train our model by:
```
sh ./Second_stage/run_bert.sh
```
In the file ``run_bert.sh``, we can modify the settings and hyper-parameters of the model.

**Note:**
1. The path of the BERT model should be set in line 86 of ``train_demo.py``.
2. The path of the pretrained encoder should be set in line 150 of ``Second_stage\fewshot_re_kit\framework.py``.


## Code
Our self-denoising FSRE model is placed in the ``self_denoise.py`` file.
