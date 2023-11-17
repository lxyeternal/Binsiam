# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi 
@File    ：loaddata.py
@Author  ：honywen
@Date    ：2023/5/31 19:30 
@Software: PyCharm
"""

import dgl
import torch
import random
from dgl import load_graphs
from AlignBatch import alignbatch


class LoadData:
    def __init__(self):
        self.all_dataset_path = "../../../dataset/DGLBatch/all_pairs_hete_test.bin"
        self.all_dataset, self.labels_tmp = load_graphs(self.all_dataset_path)
        self.all_dataset = self.all_dataset[:100]
        self.labels_tmp_new = self.labels_tmp['glabel'].tolist()[:100]
        self.all_labels = dict()
        self.all_labels['glabel'] = torch.tensor(self.labels_tmp_new)
        self.dataset_list = list()
        self.labels_list = list()
        self.graph_num_batch = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def transfer_dataset(self, dataset, lables):
        pairs = [(dataset[i], dataset[i + 1]) for i in range(0, len(dataset), 2)]
        labels = lables['glabel'].tolist()
        self.dataset_list = self.dataset_list + pairs
        self.labels_list = self.labels_list + labels

    def combine_dataset(self):
        self.transfer_dataset(self.all_dataset, self.all_labels)

    #   划分数据集：训练集60%，验证集20%,测试集20%
    def datasplit(self):
        self.combine_dataset()
        len_data = len(self.dataset_list)
        batch_num = int(len_data / self.graph_num_batch)
        dataloader = list()
        for i in range(batch_num):
            start_index = i * self.graph_num_batch
            end_index = (i + 1) * self.graph_num_batch
            tmp_graph = self.dataset_list[start_index:end_index]
            tmp_label = self.labels_list[start_index:end_index]
            tmp_graph1, tmp_graph2 = zip(*tmp_graph)
            new_tmp_graph1 = alignbatch(tmp_graph1)
            new_tmp_graph2 = alignbatch(tmp_graph2)
            batched_graph1 = dgl.batch(list(new_tmp_graph1))
            batched_graph2 = dgl.batch(list(new_tmp_graph2))
            tensor_label = torch.tensor(tmp_label).to(self.device)
            dataloader.append(([batched_graph1, batched_graph2], tensor_label))

        len_dataloader = len(dataloader)
        trainvalidaset_num = int(len_dataloader * 0.8)
        trainvalidaset_idxs = random.sample(range(0, len_dataloader), trainvalidaset_num)
        validation_num = int(trainvalidaset_num * 0.25)
        validation_idxs = random.sample(trainvalidaset_idxs, validation_num)
        trainset_idxs = list(set(trainvalidaset_idxs) - set(validation_idxs))
        train_dataloader = [dataloader[i] for i in trainset_idxs]
        validation_dataloader = [dataloader[i] for i in validation_idxs]
        test_dataloader = list()
        for i in range(len_dataloader):
            if i in trainvalidaset_idxs:
                pass
            else:
                test_dataloader.append(dataloader[i])
        random.shuffle(train_dataloader)
        random.shuffle(validation_dataloader)
        random.shuffle(test_dataloader)
        print(len(train_dataloader), len(validation_dataloader), len(test_dataloader))
        return train_dataloader, validation_dataloader, test_dataloader
