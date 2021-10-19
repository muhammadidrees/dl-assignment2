import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset

import sys
sys.path.append('./data/')
from data_preprocessing import *


class AdultDataset(Dataset):

    def __init__(self):
        adult = pd.read_csv("data/adult/adult.data")
        adult.columns = get_labels_from("data/adult/adult.columns_raw")
        
        self.scaler = MinMaxScaler()
        self.encoder = LabelEncoder()
        
        X, y = split_dataframe_into_X_y(adult)
        X = remove_leading_trailing_space_from(X)

        X = X.apply(self.encoder.fit_transform)
        y = y.apply(self.encoder.fit_transform)

        rangeValues = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

        X[rangeValues] = self.scaler.fit_transform(X[rangeValues])

        print(X.head())
        print(y.head())

        self.X = torch.from_numpy(X.values.astype(np.float32))
        self.y = torch.from_numpy(y.values.astype(np.float32))

        self.n_samples = adult.shape[0]


    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = AdultDataset()
first = dataset[0]