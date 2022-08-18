
import sys
sys.path.append('../')
from modelUtil import *
from FedUser import CDPUser, LDPUser, opacus
from FedServer import Server
import logging
import os
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
DATA_NAME = "cifar"
NUM_CLIENTS = 100
NUM_CLASSES = 10
NUM_CLASES_PER_CLIENT= 2
MODEL = "clip"
LOCAL_MODEL = "linear_model_DN"
MODE = "CDP"
EPOCHS = 1
ROUNDS = 30
BATCH_SIZE = 100
LEARNING_RATE_DIS = 5e-2
target_epsilon = 6
target_delta = 1e-4
max_norm = 0.5
sample_rate=0.5
LOG_FORMAT = "%(asctime)s - %(message)s"

user_param = {'disc_lr': LEARNING_RATE_DIS, 'epochs': EPOCHS}
server_param = {}
if MODE == "LDP":
    user_obj = LDPUser
    user_param['rounds'] = ROUNDS
    user_param['target_epsilon'] = target_epsilon
    user_param['target_delta'] = target_delta
elif MODE == "CDP":
    user_obj = CDPUser
    server_param['noise_multiplier'] = opacus.accountants.utils.get_noise_multiplier(target_epsilon=target_epsilon,
                                                                                 target_delta=target_delta, 
                                                                                 sample_rate=sample_rate, steps=ROUNDS)
    print(server_param['noise_multiplier'])
    server_param['sample_clients'] = sample_rate*NUM_CLIENTS
else:
    raise ValueError("Choose mode from [CDP, LDP]")

x_train = [np.load(f"features_{NUM_CLASES_PER_CLIENT}_classes_{MODEL}/{index}_train_x.npy") for index in range(NUM_CLIENTS)]
y_train = [np.load(f"features_{NUM_CLASES_PER_CLIENT}_classes_{MODEL}/{index}_train_y.npy") for index in range(NUM_CLIENTS)]
x_test = [np.load(f"features_{NUM_CLASES_PER_CLIENT}_classes_{MODEL}/{index}_test_x.npy") for index in range(NUM_CLIENTS)]
y_test = [np.load(f"features_{NUM_CLASES_PER_CLIENT}_classes_{MODEL}/{index}_test_y.npy") for index in range(NUM_CLIENTS)]
trainsets = [torch.utils.data.TensorDataset(torch.from_numpy(x_train[index]), torch.from_numpy(y_train[index])) for index in range(NUM_CLIENTS)]
testsets = [torch.utils.data.TensorDataset(torch.from_numpy(x_test[index]), torch.from_numpy(y_test[index])) for index in range(NUM_CLIENTS)]

train_dataloaders = [torch.utils.data.DataLoader(trainsets[index], batch_size=BATCH_SIZE, shuffle=True) for index in range(NUM_CLIENTS)]
test_dataloaders = [torch.utils.data.DataLoader(testsets[index], batch_size=BATCH_SIZE, shuffle=True) for index in range(NUM_CLIENTS)]

users = [user_obj(i, device, LOCAL_MODEL, NUM_CLASSES, train_dataloaders[i], **user_param) for i in range(NUM_CLIENTS)]
server = Server(device, LOCAL_MODEL, NUM_CLASSES, **server_param)
for i in range(NUM_CLIENTS):
    users[i].set_model_state_dict(server.get_model_state_dict())
for round in range(ROUNDS):
    random_index = np.random.choice(NUM_CLIENTS, int(sample_rate*NUM_CLIENTS), replace=False)
    for index in random_index:users[index].train()
    server.agg_updates([users[index].get_model_state_dict() for index in random_index])
    evaluate_global(users, train_dataloaders, range(NUM_CLIENTS))
    for i in range(NUM_CLIENTS):
        users[i].set_model_state_dict(server.get_model_state_dict())
    print(f"Round: {round+1}")
    evaluate_global(users, train_dataloaders, range(NUM_CLIENTS))
    evaluate_global(users, test_dataloaders, range(NUM_CLIENTS))
    if round == ROUNDS-1:
        torch.save(weights_agg, f'weights/{DATA_NAME}_{MODEL}_{MODE}.pth')
    if MODE == "LDP":
        eps = max([user.epsilon for user in users])
        print(f"Epsilon: {eps}")
