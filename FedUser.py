from modelUtil import *
from collections import OrderedDict
import torchmetrics
import opacus
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
import time

class CDPUser:
    def __init__(self, index, device, model, n_classes, train_dataloader, epochs, max_norm=1.0, disc_lr=5e-3, flr = 1e-1):
        self.index = index
        self.model = globals()[model](num_classes=n_classes)
        self.train_dataloader = train_dataloader
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.disc_lr = disc_lr
        self.acc_metric = torchmetrics.Accuracy().to(device)
        self.device = device
        self.max_norm= max_norm
        self.epochs = epochs
        self.flr = flr
        self.agg = True
        if "IN" in model:
            self.optim = torch.optim.SGD([
                                            {'params': self.model.norm.parameters(), 'lr': self.flr},
                                            {'params': [v for k, v in self.model.named_parameters() if "norm" not in k]}], lr=self.disc_lr)
            self.agg = False
        else:
            self.optim = torch.optim.SGD(self.model.parameters(), self.disc_lr)
        # self.optim = torch.optim.SGD(self.model.parameters(), self.disc_lr)

    def train(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            losses = []
            for images, labels in self.train_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optim.zero_grad()
                logits, preds = self.model(images)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optim.step()
                self.acc_metric(preds, labels)
                losses.append(loss.item())
            print(f"Client: {self.index} ACC: {self.acc_metric.compute()}, Loss:{np.mean(losses)}")
            self.acc_metric.reset()
        self.model.to('cpu')
        # print(f"{self.index} finished at {time.strftime('%X')}")

    def evaluate(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        testing_corrects = 0
        testing_sum = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                _, preds = self.model(images)
                testing_corrects += torch.sum(torch.argmax(preds, dim=1) == labels)
                testing_sum += len(labels)
        self.model.to('cpu')
        return testing_corrects.cpu().detach().numpy(), testing_sum

    def get_model_state_dict(self):
        return self.model.state_dict()

    def set_model_state_dict(self, weights):
        if self.agg == False:
            for key, value in self.model.state_dict().items():
                if 'norm' not in key and 'bn' not in key and 'downsample.1' not in key:
                    self.model.state_dict()[key].data.copy_(weights[key])
        else:
            for key, value in self.model.state_dict().items():
                self.model.state_dict()[key].data.copy_(weights[key])

class LDPUser(CDPUser):
    def __init__(self, index, device, model, n_classes, train_dataloader, epochs, rounds, target_epsilon, target_delta, max_norm=2.0, disc_lr=5e-1):
        super().__init__(index, device, model, n_classes, train_dataloader, epochs=epochs, max_norm=max_norm, disc_lr=disc_lr)
        self.rounds = rounds
        self.target_epsilon = target_epsilon
        self.epsilon = 0
        self.delta = target_delta
        self.model = ModuleValidator.fix(self.model)
        self.optim = torch.optim.SGD(self.model.parameters(), self.disc_lr)
        self.make_local_private()
        self.agg = True
        if "IN" in model:
            self.agg = False

    def make_local_private(self):
        self.privacy_engine = opacus.PrivacyEngine()
        self.model, self.optim, self.train_dataloader = self.privacy_engine.make_private_with_epsilon(module=self.model, optimizer=self.optim, 
                                                                                                      data_loader=self.train_dataloader, epochs=self.epochs*self.rounds, 
                                                                                                      target_epsilon=self.target_epsilon, target_delta=self.delta, 
                                                                                                      max_grad_norm=self.max_norm)
    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            with BatchMemoryManager(data_loader=self.train_dataloader, max_physical_batch_size=3, optimizer=self.optim) as batch_loader:
                for images, labels in batch_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optim.zero_grad()
                    logits, preds = self.model(images)
                    loss = self.loss_fn(logits, labels)
                    loss.backward()
                    self.optim.step()
                    self.acc_metric(preds, labels)
        self.epsilon = self.privacy_engine.get_epsilon(self.delta)
        print(f"Client: {self.index} ACC: {self.acc_metric.compute()}, episilon: {self.epsilon}")
        self.acc_metric.reset()
        self.model.to('cpu')
