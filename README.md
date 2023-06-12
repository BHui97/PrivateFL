# PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation

This is the Pytorch implementaion of our paper, PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation.

## Experiment Setup

We use miniconda to create a virtual environment with python 3.8, you can install miniconda use the following script if you are using Linux-x86-64bit machine:

(Optional for install miniconda)
```bash
cd script &&
bash install_conda.sh
```

Then use the following script to download the requirements:
```bash
cd script &&
bash setup.sh
```

## Code Usage
### Train from scratch
You can use the following script to train from scratch. 

```bash
cd script &&
bash train.sh
```

You can also change the parameters in script/train.sh, e.g., --data --nclient --nclass --ncpc --model --mode --round --epsilon --sr --lr, following the choices listed in parse_arguments() of FedAverage.py. The value of the parameters can be found in our paper.


### Train with frozen encoder
You can choose encoder from [ResNeXt, SimCLR, CLIP] to extract the features.
#### Feature extraction
```
cd transfer && python extract_feature.py
```

#### Train one-layer classifier
You can use 'linear_model', 'linear_model_DN" and 'linear_model_DN_IN" as FedAVG, FedAVG+DN, and FedAVG+DN+IN.
```
python FedTransfer.py
```
Please reduce the value of 'max_physical_batch_size' in BatchMemoryManager (FedUser/LDPUser) if facing CUDA out of memory.
