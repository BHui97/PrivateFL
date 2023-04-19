from modelUtil import *
from datasets import *
from FedUser import CDPUser, LDPUser, opacus
from FedServer import LDPServer, CDPServer
from datetime import date
import logging
import os
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
today = date.today().isoformat()
DATA_NAME = "cifar"
NUM_CLIENTS = 100
NUM_CLASSES = 10
NUM_CLASES_PER_CLIENT= 2
MODEL = sys.argv[1]
MODE = "LDP"
EPOCHS = 1
ROUNDS = int(sys.argv[2])
BATCH_SIZE = 64
LEARNING_RATE_DIS = 1e-1
target_epsilon = int(sys.argv[3])
target_delta = 1e-3
sample_rate=1

user_param = {'disc_lr': LEARNING_RATE_DIS, 'epochs': EPOCHS}
server_param = {}
if MODE == "LDP":
    user_obj = LDPUser
    server_obj = LDPServer
    user_param['rounds'] = ROUNDS
    user_param['target_epsilon'] = target_epsilon
    user_param['target_delta'] = target_delta
elif MODE == "CDP":
    user_obj = CDPUser
    server_obj = CDPServer
    server_param['noise_multiplier'] = opacus.accountants.utils.get_noise_multiplier(target_epsilon=target_epsilon,
                                                                                 target_delta=target_delta, 
                                                                                 sample_rate=sample_rate, steps=ROUNDS)
    print(server_param['noise_multiplier'])
    server_param['sample_clients'] = sample_rate*NUM_CLIENTS
else:
    raise ValueError("Choose mode from [CDP, LDP]")

train_dataloaders, test_dataloaders = gen_random_loaders(DATA_NAME, '~/torch_data', NUM_CLIENTS,
                                                         BATCH_SIZE, NUM_CLASES_PER_CLIENT, NUM_CLASSES)
users = [user_obj(i, device, MODEL, NUM_CLASSES, train_dataloaders[i], **user_param) for i in range(NUM_CLIENTS)]
server = server_obj(device, MODEL, NUM_CLASSES, **server_param)
for i in range(NUM_CLIENTS):
    users[i].set_model_state_dict(server.get_model_state_dict())
for round in range(ROUNDS):
    random_index = np.random.choice(NUM_CLIENTS, int(sample_rate*NUM_CLIENTS), replace=False)
    for index in random_index:users[index].train()
    if MODE == "LDP":
        weights_agg = agg_weights([users[index].get_model_state_dict() for index in random_index])
        for i in range(NUM_CLIENTS):
            users[i].set_model_state_dict(weights_agg)
    else:
        server.agg_updates([users[index].get_model_state_dict() for index in random_index])
        for i in range(NUM_CLIENTS):
            users[i].set_model_state_dict(server.get_model_state_dict())
    print(f"Round: {round+1}")
    evaluate_global(users, test_dataloaders, range(NUM_CLIENTS))
    if MODE == "LDP":
        eps = max([user.epsilon for user in users])
        print(f"Epsilon: {eps}")
