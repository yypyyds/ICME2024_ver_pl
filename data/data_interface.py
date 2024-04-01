import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from data.CASdata import CASDeltaDataset, unlabeled_CASDeltaDataset, valdataset
from data.TAUdata import DeltaDataset



class DInterface(pl.LightningDataModule):

    def __init__(self, num_workers=8,
                 dataset='',
                 **kwargs):
        """A dataloader wrapper for both supervised learning and unsupervised learning, including TAU and CAS datasets.

        Args:
            num_workers (int, optional): a args for LightningDataModule.
            dataset (str, optional): Dataset to load, TAU or CAS. Defaults to ''.
            train_val_ratio (_type_, optional): the ratio between train data and validate data, set it if in training stage. Defaults to None.
        """
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.csv_file = kwargs['csv_file']
        self.fea_path = kwargs['fea_path']
        if kwargs['train_val_ratio'] is not None:
            train_num = int(len(self.csv_file) * kwargs['train_val_ratio'])
            self.train_csv = self.csv_file[:train_num]
            self.val_csv = self.csv_file[train_num:]
        else:
            self.test_csv = self.csv_file
        if 'unlabel_csv' in kwargs.keys():
            self.unlabel_csv = kwargs['unlabel_csv']
        
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if self.dataset == 'TAU':
            if stage == 'fit':
                self.trainset = DeltaDataset(self.train_csv, self.fea_path)
                self.valset = DeltaDataset(self.val_csv, self.fea_path)
            if stage == 'test':
                self.testset = DeltaDataset(self.test_csv, self.fea_path)
                
        if self.dataset == 'CAS':
            if stage == 'fit':
                self.trainset = CASDeltaDataset(self.train_csv, self.fea_path)
                self.valset = valdataset(self.val_csv, self.fea_path)
                self.unlabelset = unlabeled_CASDeltaDataset(self.unlabel_csv, self.fea_path)
                self.iteration = self.unlabelset.__len__()//self.batch_size
            if stage == 'test':
                self.testset = valdataset(self.test_csv, self.fea_path)
        


    def train_dataloader(self):
        if self.dataset == 'TAU':
            return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        else:
            dataset1 = DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            dataset2 = DataLoader(self.unlabelset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            loaders = {
                'label': dataset1,
                'unlabel': dataset2
            }
            return loaders

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)