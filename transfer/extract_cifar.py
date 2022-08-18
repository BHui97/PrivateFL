import sys
sys.path.append('../')
from modelUtil import *
from datasets import *
from FedUser import CDPUser, LDPUser, opacus
from FedServer import Server
from datetime import date
import logging
import os
from pl_bolts.models.self_supervised import SimCLR
from sklearn.linear_model import LogisticRegression
from resnext import resnext
from CLIP import clip
DATA_NAME = "cifar"
NUM_CLIENTS = 100
NUM_CLASSES = 10
NUM_CLASES_PER_CLIENT= 2
MODEL = "clip"
BATCH_SIZE = 64
preprocess = None
if MODEL == "simclr":
    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
    model = simclr.encoder
    model.eval()
elif MODEL == "resnext":
    model = resnext(cardinality=8, num_classes=100, depth=29, widen_factor=4, dropRate=0,)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.eval()
    checkpoint = torch.load("resnext-8x64d/model_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.module.classifier = torch.nn.Identity()
elif MODEL == "clip":
    model, preprocess = clip.load('ViT-B/32', 'cpu')
    model = model.encode_image
print('model loaded')

train_dataloaders, test_dataloaders = gen_random_loaders(DATA_NAME, '~/torch_data', NUM_CLIENTS,
                                                         BATCH_SIZE, NUM_CLASES_PER_CLIENT, NUM_CLASSES, preprocess)
os.makedirs(f'features_{NUM_CLASES_PER_CLIENT}_classes_{MODEL}/', exist_ok=True)
for index in range(NUM_CLIENTS):
    features_train = []
    y_train = []
    for x, y in train_dataloaders[index]:
        features = model(x)
        features_train.append(features.cpu().detach().numpy())
        y_train.append(y)
    features_train = np.concatenate(features_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    np.save(f'features_{NUM_CLASES_PER_CLIENT}_classes_{MODEL}/{index}_train_x.npy', features_train)
    np.save(f'features_{NUM_CLASES_PER_CLIENT}_classes_{MODEL}/{index}_train_y.npy', y_train)
    features_test = []
    y_test = []
    for x, y in test_dataloaders[index]:
        features = model(x)
        features_test.append(features.cpu().detach().numpy())
        y_test.append(y)
    features_test = np.concatenate(features_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    np.save(f'features_{NUM_CLASES_PER_CLIENT}_classes_{MODEL}/{index}_test_x.npy', features_test)
    np.save(f'features_{NUM_CLASES_PER_CLIENT}_classes_{MODEL}/{index}_test_y.npy', y_test)