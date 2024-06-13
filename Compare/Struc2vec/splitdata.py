# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# @File     : splitdata.py
# @Project  : Binsiam
# Time      : 2023/12/31 01:34
# Author    : honywen
# version   : python 3.8
# Description：
"""

import torch
import pickle
import random
from torch_geometric.data import Data, DataLoader


class LoadData:
    def __init__(self):
        self.all_dataset_path = "/Users/blue/Documents/Binsiam/Compare/Struc2vec/data/findutils-4.9.0-False-True-False.bin"
        with open(self.all_dataset_path, 'rb') as file:
            data = pickle.load(file)
        self.all_dataset = data['functions']
        self.all_labels = data['labels']
        self.dataset_list = list()
        self.labels_list = list()
        self.graph_num_batch = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def transfer_dataset(self, dataset, labels):
        pairs = [(dataset[i], dataset[i + 1]) for i in range(0, len(dataset), 2)]
        # labels = lables['labels'].tolist()
        self.dataset_list = self.dataset_list + pairs
        self.labels_list = self.labels_list + labels

    def transform_lists(self, dataset, labels):
        # 分离 a 列表为两个子列表，一个对应 b 中的 0，另一个对应 1
        dataset_pos = [dataset[i] for i in range(len(dataset)) if labels[i] == 0]
        dataset_neg = [dataset[i] for i in range(len(dataset)) if labels[i] == 1]
        # 重新组合 a_new，交替从 a_0 和 a_1 中取元素
        min_length = min(len(dataset_neg), len(dataset_pos))
        new_dataset = [None] * (2 * min_length)
        new_dataset[::2] = dataset_pos[:min_length]
        new_dataset[1::2] = dataset_neg[:min_length]
        # 创建对应的 b_new
        new_labels = torch.tensor([0, 1] * min_length, dtype=torch.float32)
        return new_dataset, new_labels

    def combine_dataset(self):
        self.transfer_dataset(self.all_dataset, self.all_labels)
        self.dataset_list, self.labels_list = self.transform_lists(self.dataset_list, self.labels_list)

    def pair_data_with_labels(self, data, label):
        paired_data = []
        for i in range(0, len(data), 2):
            paired_data.append(((data[i], data[i + 1]), label[i // 2]))
        return paired_data


    def split_data(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        self.combine_dataset()
        paired_data = self.pair_data_with_labels(self.dataset_list, self.labels_list)
        # Shuffle data
        random.shuffle(paired_data)
        # Calculate split sizes
        total_size = len(paired_data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        # Split data
        train_data = paired_data[:train_size]
        val_data = paired_data[train_size:train_size + val_size]
        test_data = paired_data[train_size + val_size:]
        # Create DataLoaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

        return train_loader, val_loader, test_loader