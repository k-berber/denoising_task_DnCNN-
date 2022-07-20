import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_clean_data_paths(path: str):
    clean = 'clean/'
    list_clean = []
    with os.scandir(path+clean) as entries:
        for entry in entries:
            if entry.is_dir():
                list_clean.append(entry.path)
    return sorted(list_clean)


def get_data_from_clean_data_paths(clean_data_paths: list):
    """
    input: paths to clean data
    output: loaded clean and noisy data to lists of np.arrays
    """
    list_of_clean_data = []
    list_of_noisy_data = []

    cnt = 0
    # print(len(clean_data_paths))
    for path in clean_data_paths:
        cnt += 1
        if cnt % 10 == 0:
            print(cnt, path)

        with os.scandir(path) as entries:
            for input in entries:
                if input.is_file():
                    clean_path = input.path
                    noisy_path = input.path.replace('clean', 'noisy', 1)

                    list_of_clean_data.append(np.load(clean_path).T)
                    list_of_noisy_data.append(np.load(noisy_path).T)
    return list_of_clean_data, list_of_noisy_data

class Detect(Dataset):
    """
      data - clean and noisy data
      targets - noise residuals of the same size as noisy or clean data
    """

    def __init__(self, clean, noisy):
        # initialize class object

        # concatenate clean and noisy data lists
        self.data = noisy
        # generate corresponding targets
        self.targets = [noisy[i] - clean[i] for i in range(len(clean))]

    def __len__(self):
        # standard interface function for Dataset
        if self.data != None:
            return len(self.data)

    def __getitem__(self, idx):
        # standard interface function for Dataset
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.data[idx], self.targets[idx]]  # {'mel_data': self.data[idx], 'target': self.targets[idx]}
        return sample


def get_DataLoader(clean_data_list, noisy_data_list, batch_size, shuffle=True):
    dataset = Detect(clean_data_list, noisy_data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=match_function, pin_memory=True)
    return dataloader

