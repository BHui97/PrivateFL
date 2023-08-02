# PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation

This is the Pytorch implementation of our paper, PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation.

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

Run the following script to extract features from [ResNeXt, SimCLR, CLIP] and train a one-layer classifier, you may need to download ResNext using [this link](https://nam02.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1v-ZOPhSHaP1DGygMn4B3AHBKKiQRb4sz%2Fview%3Fusp%3Ddrive_link&data=05%7C01%7Cyc.yang%40jhu.edu%7C1fcd39208359468a92cf08db9348949d%7C9fa4f438b1e6473b803f86f8aedf0dec%7C0%7C0%7C638265712203751267%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=ZJLMpgA%2F0GUB2DwIhmPSjNsmSBUbGT20nSzI9PIAB6k%3D&reserved=0):

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


