import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import os


mnist_transform = transforms.Compose([transforms.ToTensor()])
# class Femnist(Dataset):
#     def __init__(self, index, root='/mnt/HDD/torch_data/femnist', digits_only = True, train=True, transform=mnist_transform):
#         if index >3579:
#             raise ValueError("choose index from range(0, 3579)")
#         self.root = os.path.join(root, "train" if train else "test")
#         files_name = os.listdir(self.root)
#         self.imgs, self.targets = torch.load(os.path.join(self.root, files_name[index])).values()
#         self.transform = transform
#         self.digits_only = digits_only
#         self.file_name = files_name[index]
#         if self.digits_only:
#             self.digits_index = np.where(self.targets<10)[0]
#             self.length = self.digits_index.size
#     def __getitem__(self, index):
#         if self.digits_only:
#             index = self.digits_index[index]
#         img = self.imgs[index]
#         print(self.targets[index])
#         print(img)
#         return img, self.targets[index]

#     def __len__(self):
#         return self.length

class Femnist(Dataset):
    def __init__(self, index, root='/mnt/HDD/leaf/data/femnist/data', digits_only = True, train=True, transform=mnist_transform):
        if index >3579:
            raise ValueError("choose index from range(0, 3579)")
        self.root = os.path.join(root, "train" if train else "test")
        files_name = os.listdir(self.root)
        self.imgs, self.targets = torch.load(os.path.join(self.root, files_name[index])).values()
        self.transform = transform
        self.digits_only = digits_only
        self.file_name = files_name[index]
        if self.digits_only:
            self.digits_index = np.where(self.targets<10)[0]
            self.length = self.digits_index.size
    def __getitem__(self, index):
        if self.digits_only:
            index = self.digits_index[index]
        img = self.imgs[index]
        print(self.targets[index])
        print(img)
        return img, self.targets[index]

    def __len__(self):
        return self.length