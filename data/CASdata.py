import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset


selected_scene_list = [
    "Bus",
    "Airport",
    "Metro",
    "Restaurant",
    "Shopping mall",
    "Public square",
    "Urban park",
    "Traffic street",
    "Construction site",
    "Bar",
]
class_2_index = {
    "Bus": 0,
    "Airport": 1,
    "Metro": 2,
    "Restaurant": 3,
    "Shopping mall": 4,
    "Public square": 5,
    "Urban park": 6,
    "Traffic street": 7,
    "Construction site": 8,
    "Bar": 9,
}

index_2_class = {
    0: "Bus",
    1: "Airport",
    2: "Metro",
    3: "Restaurant",
    4: "Shopping mall",
    5: "Public square",
    6: "Urban park",
    7: "Traffic street",
    8: "Construction site",
    9: "Bar",
}

# [C, F, T]
def deltas(X_in):
    X_out = (X_in[:,2:]-X_in[:,:-2])/10.0
    X_out = X_out[:,1:-1]+(X_in[:,4:]-X_in[:,:-4])/5.0
    return X_out


class CASDeltaDataset(Dataset):
    def __init__(self, csv_file, feature_path):
        self.stats_csv = csv_file
        self.root_path = feature_path
        self.file_list = []
        self.label_list = []
        self.get_file_list()

        self.cup = [[] for _ in range(10)]
        for idx, row in csv_file.iterrows():
            self.cup[class_2_index[row["scene_label"]]].append(row["filename"])

    def get_file_list(self):
        selected_data = self.stats_csv[
            self.stats_csv["scene_label"].isin(selected_scene_list)
        ]

        for index, row in selected_data.iterrows():
            label_str = row["scene_label"]

            if isinstance(label_str, str):
                filename = row["filename"]
                file_path = os.path.join(self.root_path, filename + ".npy")
                self.file_list.append(file_path)
                self.label_list.append(class_2_index[label_str])
            else:
                continue  

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        file = np.load(file_path)
        data_delta = deltas(file)
        data_delta_delta = deltas(data_delta)
        data = np.array([file[:,4:-4], data_delta[:,2:-2], data_delta_delta])
        data = torch.Tensor(data)
        
        label = self.label_list[idx]
        file_name2 = random.choice(self.cup[label])
        file_path2 = os.path.join(self.root_path, file_name2 + ".npy")
        file2 = np.load(file_path2)
        data_d2 = deltas(file2)
        data_dd2 = deltas(data_d2)
        data2 = np.array([file2[:,4:-4], data_d2[:,2:-2], data_dd2])
        data2 = torch.Tensor(data2)
        l = random.random()
        data = l*data + (1-l)*data2
        return data, label


class unlabeled_CASDeltaDataset(Dataset):
    def __init__(self, csv_file, feature_path):
        self.stats_csv = csv_file
        self.root_path = feature_path
        self.file_list = []
        self.get_file_list()

    def get_file_list(self):
        for index, row in self.stats_csv.iterrows():
            label_str = row["scene_label"]

            if not isinstance(label_str, str):
                filename = row["filename"]
                file_path = os.path.join(self.root_path, filename + ".npy")
                self.file_list.append(file_path)
            else:
                continue

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_path, file_name)
        file = np.load(file_path)
        data_delta = deltas(file)
        data_delta_delta = deltas(data_delta)
        data = np.array([file[:,4:-4], data_delta[:,2:-2], data_delta_delta])
        data = torch.Tensor(data)
        return data
    

class valdataset(Dataset):
    def __init__(self, csv_file, feature_path):
        self.stats_csv = csv_file
        self.root_path = feature_path
        self.file_list = []
        self.label_list = []
        self.get_file_list()

        self.cup = [[] for _ in range(10)]
        for idx, row in csv_file.iterrows():
            self.cup[class_2_index[row["scene_label"]]].append(row["filename"])


    def get_file_list(self):
        selected_data = self.stats_csv[
            self.stats_csv["scene_label"].isin(selected_scene_list)
        ]

        for index, row in selected_data.iterrows():
            label_str = row["scene_label"]

            if isinstance(label_str, str):
                filename = row["filename"]
                file_path = os.path.join(self.root_path, filename + ".npy")
                self.file_list.append(file_path)
                self.label_list.append(class_2_index[label_str])
            else:
                continue  

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        file = np.load(file_path)
        data_delta = deltas(file)
        data_delta_delta = deltas(data_delta)
        data = np.array([file[:,4:-4], data_delta[:,2:-2], data_delta_delta])
        data = torch.Tensor(data)
        label = self.label_list[idx]
        return data, label