import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class HandKeypointDataset(Dataset):
    def __init__(self, csv_file, data_dir, label_map):
        self.data_frame = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.label_map = label_map

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        video_id = row['video_id']
        label = row['label']
        frames = np.load(os.path.normpath(os.path.join(self.data_dir, f'{video_id}.npy')))
        label_idx = self.label_map[label]
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)


# 标签编码
labels = pd.read_csv('data/archive/Train.csv')['label'].unique()
label_map = {label: idx for idx, label in enumerate(labels)}


# 创建数据集和数据加载器
train_dataset = HandKeypointDataset('data/archive/Train.csv', 'data/processed_train', label_map)
validation_dataset = HandKeypointDataset('data/archive/Validation.csv', 'data/processed_validation', label_map)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=4)
