# Speech Enhancement Based On Deep Neural Networks

### Code reference: https://github.com/ruizhecao96/CMGAN

### Paper: https://ieeexplore.ieee.org/document/10508391

### How to start:

**1. Prepare the necessary libraries:** 
`pip install -r requirements.txt`

**2. Download VCTK-DEMAND dataset with *16 kHz*, change the dataset dir:**
You can download the dataset on: 

https://drive.google.com/file/d/1pGV79T3k030f6uc2SbUpuNhfovtmLJxN/view
```
-VCTK-DEMAND/
  -train/
    -noisy/
    -clean/
  -test/
    -noisy/
    -clean/
```
If you want to use different datasets, please ensure that you maintain this structure.

**3. For training:**
```
python train.py --epochs 50 --batch_size 4 --log_interval 500 --decay_epoch 12 --init_lr 0.0005 --cut_len 32000 --loss_weights 0.3 0.7 1 0.01 --data_dir "path for your dataset" --save_model_dir "path for saving models" --attn1 "attention block: se/cbam/eca/simam/None"
```

**4. Evaluate the checkpoints:**
```
python evaluation.py --test_dir "path of test set, for example, ./DEMAND_16KHz/test" --model_path "path of evaluated model" --save_dir "path for saving results" --attn1 "attention block: se/cbam/eca/simam/None"
```

### Project outline:
* -AudioSamples: Clean and noisy audio samples.
* -best_ckpt: Best checkpoints in the project.
* -data: Code for loading and processing data.
* -models: The components of the CMGAN model and the attention module.
* -results: Results obtained from the experiment.
* -tools: Code used to calculate evaluation metrics.
* -eval.sh: Command for evaluation.
* -evaluation.py: Code for implementing evaluation.
* -requirements.txt: Configure the environment required for the project.
* -train_gpu1.sh: Command for training.
* -train.py: Code for implementing training.
* -utils.py: Tools for data processing and weight initialisation.
 