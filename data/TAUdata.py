import os
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset


selected_scene_list = [
    "bus",
    "airport",
    "metro",
    "tram",
    "shopping_mall",
    "public_square",
    "park",
    "street_traffic",
    "street_pedestrian",
    "metro_station",
]
class_2_index = {
    "bus": 0,
    "airport": 1,
    "metro": 2,
    "tram": 3,
    "shopping_mall": 4,
    "public_square": 5,
    "park": 6,
    "street_traffic": 7,
    "street_pedestrian": 8,
    "metro_station": 9,
}

index_2_class = {
    0: "bus",
    1: "airport",
    2: "metro",
    3: "tram",
    4: "shopping_mall",
    5: "public_square",
    6:  "park",
    7: "street_traffic",
    8: "street_pedestrian",
    9: "metro_station",
}
# [C, F, T]
def deltas(X_in):
    X_out = (X_in[:,2:]-X_in[:,:-2])/10.0
    X_out = X_out[:,1:-1]+(X_in[:,4:]-X_in[:,:-4])/5.0
    return X_out


class DeltaDataset(Dataset):
    def __init__(self, csv_file, feature_path):
        self.stats_csv = csv_file
        self.root_path = feature_path
        self.file_list = []
        self.label_list = []
        self.get_file_list()

    def get_file_list(self):
        selected_data = self.stats_csv[
            self.stats_csv["scene_label"].isin(selected_scene_list)
        ]

        for index, row in selected_data.iterrows():
            label_str = row["scene_label"]

            if isinstance(label_str, str):
                filename = row["filename"][6:-4]
                file_path = os.path.join(self.root_path, filename + ".npy")
                self.file_list.append(file_path)
                self.label_list.append(class_2_index[label_str])
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
        label = self.label_list[idx]
        return data, label
