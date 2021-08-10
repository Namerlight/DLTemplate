import json
import os

import numpy as np
from torch.utils.data import Dataset


class json_loader(Dataset):
    def __init__(self, data, labels, transforms=None):

        # self.loc = data_loc
        # self.X = os.listdir(data_loc)

        self.X = data
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):

        data = self.X[i].astype(np.float32)
        label = self.y[i]

        if self.transforms:
            data = self.transforms(self.X)

        if self.y is not None:
            return data, label

        else:
            return data, label


def get_dataset(train_data_loc, test_data_loc):

    train_all_data = []
    train_all_labels = []
    test_all_data = []
    test_all_labels = []

    all_files = os.listdir(train_data_loc)

    for i in range(len(all_files)):
        if all_files[i][:3] == 'zin':
            label = 0
            train_all_labels.append(label)
        else:
            label = 1
            train_all_labels.append(label)

        with open(train_data_loc + '/' + all_files[i]) as json_file:
                data = json.load(json_file)
                data_arr = np.array(data["embedding"]).astype(np.uint8)
                train_all_data.append(data_arr)

    all_files = os.listdir(test_data_loc)

    for i in range(len(all_files)):

        if all_files[i][:3] == 'zin':
            label = 0
            test_all_labels.append(label)
        else:
            label = 1
            test_all_labels.append(label)

        with open(test_data_loc + '/' + all_files[i]) as json_file:
            data = json.load(json_file)
            data_arr = np.array(data["embedding"]).astype(np.uint8)
            test_all_data.append(data_arr)

    train_dataset = json_loader(data=train_all_data, labels=train_all_labels)
    test_dataset = json_loader(data=test_all_data, labels=test_all_labels)

    return train_dataset, test_dataset

