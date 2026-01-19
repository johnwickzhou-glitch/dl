import torch.utils.data as data
from io import open
from os import path
import numpy as np
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils_math import *


class HDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 csv_file: str = './data/data.csv',
                 sample_size: int = 30,
                 horizon: int = 1080,
                 ):
        """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            """

        self.csv_file = csv_file
        self.sample_size = sample_size
        self.horizon = horizon
        # store X as a list, each element is a 100*42(len * attributes num) np array [velx;vely;x;y;acc;angle] * 7
        self.X_frames = []
        # store Y as a list, each element is a 100*4(len * attributes num) np array[velx;vely;x;y]
        self.Y_frames = []
        self.ground_truth = []

        self.get_XY()
        self.tensor_data()

    def __len__(self):
        return len(self.X_frames)

    def __getitem__(self, idx):
        single_data = self.X_frames[idx]
        first_label = self.Y_frames[idx]

        return (single_data, first_label)

    def get_ground_truth(self):
        ground_truth = self.ground_truth[0]
        return ground_truth

    def get_XY(self):
        dataS = pd.read_csv('./data/data.csv')
        dataS = dataS.drop(['Unnamed: 0', 'Unnamed: 32'], axis=1)
        dataS_cleaned = dataS.dropna()
        data_array = dataS_cleaned.to_numpy().T
        data_array = np.delete(data_array, -1, axis=1)
        data_array = data_array.flatten()
        print(data_array.shape)
        # delta_t=0.2*60=2 min
        reshaped_array = data_array.reshape(int(data_array.shape[0] / 60), 60).T
        new_array = reshaped_array.flatten()
        self.ground_truth=self.ground_truth+[new_array[0:360]]
        for i in range(int(new_array.shape[0] / 360)):
            # i,N,M,M
            self.X_frames = self.X_frames + [new_array[i * 360:(i + 1) * 360]]
            self.Y_frames = self.Y_frames + [new_array[i * 360:(i + 1) * 360]]

    def tensor_data(self):
        self.X_frames = [torch.tensor(item) for item in self.X_frames]
        self.Y_frames = [torch.tensor(item) for item in self.Y_frames]


def get_dataset(BatchSize):
    '''
    return torch.util.data.Dataloader for train,test and validation
    '''
    filename = './data_prepared/data_' + str(360) + '.pickle'
    if path.exists(filename):
        with open(filename, 'rb') as data:
            dataset = pickle.load(data)
    else:
        dataset = HDataset()
        with open(filename, 'wb') as output:
            pickle.dump(dataset, output)
    # split dataset into train test and validation 7:2:1
    num_train = (int)(dataset.__len__() * 0.7)
    num_test = (int)(dataset.__len__()) - num_train
    train, test = torch.utils.data.random_split(dataset, [num_train, num_test])
    train_loader = DataLoader(train, batch_size=BatchSize, shuffle=True)
    test_loader = DataLoader(test, batch_size=BatchSize, shuffle=True)
    return (train_loader, test_loader, dataset)
