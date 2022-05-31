import random
from torch.utils import data

import Consts

seed_size = Consts.seed_size

class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        X = self.X[index].float().reshape(-1)
        Y = self.Y[index].float();
        return X, Y

    def getRandomX(self):
        index = random.randint(0, len(self.X) - 1)
        return self.X[index][0:seed_size].float().reshape(-1)