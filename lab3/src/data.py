import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch_geometric
import torch_geometric.datasets
from torch_geometric.transforms import NormalizeFeatures
import os


TRAIN_FLAG = 0
VAL_FLAG = 1
TEST_FLAG = 2
DEFAULT_FLAG = 3


class Dataset(Dataset):
    def __init__(self, path, flag=DEFAULT_FLAG, dataset_name='Cora'):
        super(Dataset, self).__init__()
        self.path = path
        self.flag = flag
        self.dataset_name = dataset_name
        self.data, self.label = self.load_data()

    def load_data(self):
        if self.dataset_name == 'Cora':
            dataset = torch_geometric.datasets.Planetoid(root=self.path,
                                                         name='Cora', transform=NormalizeFeatures())[0]
        elif self.dataset_name == 'Citeseer':
            dataset = torch_geometric.datasets.Planetoid(root=self.path,
                                                         name='Citeseer', transform=NormalizeFeatures())[0]
        elif self.dataset_name == 'Pubmed':
            dataset = torch_geometric.datasets.Planetoid(root=self.path,
                                                         name='Pubmed', transform=NormalizeFeatures())[0]
        else:
            raise ValueError('Unknown dataset name.')
        if self.flag == TRAIN_FLAG:
            data = dataset.x[dataset.train_mask]
            label = dataset.y[dataset.train_mask]
        elif self.flag == VAL_FLAG:
            data = dataset.x[dataset.val_mask]
            label = dataset.y[dataset.val_mask]
        elif self.flag == TEST_FLAG:
            data = dataset.x[dataset.test_mask]
            label = dataset.y[dataset.test_mask]
        else:
            data = dataset.x
            label = dataset.y
        return data, label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data_loader(self, batch_size):
        if self.flag == 3:
            raise ValueError('flag should be 0, 1, 2.')
        return DataLoader(self, batch_size=batch_size, shuffle=(self.flag == TRAIN_FLAG))

    # batch_size = 16
    # cur_dir = os.path.dirname(__file__)
    # dataset_dir = os.path.join(cur_dir, '../dataset/')

    # train_loader = CoraDataset(dataset_dir, TRAIN_FLAG).get_data_loader(batch_size)
    # val_loader = CoraDataset(dataset_dir, VAL_FLAG).get_data_loader(batch_size)
    # test_loader = CoraDataset(dataset_dir, TEST_FLAG).get_data_loader(batch_size)

    # loader0 = next(iter(train_loader))
    # print(loader0[0].shape)
