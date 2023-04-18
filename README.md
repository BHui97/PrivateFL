# PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation

This is Pytorch implementaion of our paper, PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation.

## Code Usage
### Train from scratch
You can choose model from modelUtil.py, set #round and #epsilon. You can also modify the parameters like #NUM_CLIENTS,#NUM_CLASSES_PER_CLIENT, #Delta, etc. in FedAverage.py.

```bash
python FedAverage.py [model_name] [round] [episilon]
```


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
