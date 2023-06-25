# PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation

This is the Pytorch implementaion of our paper, PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation.

## Experiment Setup

First enter the following path:

```bash
cd script
```

We use miniconda to create a virtual environment with python 3.8, you can install miniconda use the following script if you are using Linux-x86-64bit machine:

(Optional for install miniconda)
```bash
bash install_conda.sh
```

Then use the following script to download the requirements:
```bash
bash setup.sh
```

## Code Usage
### Train from scratch
You can use the following script to train from scratch. 

```bash
bash fedavg.sh
```

You can also change the parameters in script/train.sh, e.g., --data --nclient --nclass --ncpc --model --mode --round --epsilon --sr --lr, following the choices listed in parse_arguments() of FedAverage.py. The value of the parameters can be found in our paper.


### Train with frozen encoder

Run the following script to extract features from [ResNeXt, SimCLR, CLIP] and train a one-layer classifier:

```bash
bash fedtransfer.sh
```
Please reduce the value of --physical_bs if facing CUDA out of memory.

## Citation

```
@inproceedings{yangprivatefl,
  title={PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation},
  author={Yang, Yuchen and Hui, Bo and Yuan, Haolin and Gong, Neil and Cao, Yinzhi}
  booktitle = {Proceedings of the USENIX Security Symposium (Usenix'23)},
  year = {2023}
}
```


