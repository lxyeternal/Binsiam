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


class LoadData:
    def __init__(self):
        self.dataset_path = "../../dataset/DGLBatch/test_pairs.bin"
        self.dataset, self.labels = load_graphs(self.dataset_path)
        self.dataset_list = list()
        self.labels_list = list()
        self.graph_num_batch = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def transfer_dataset(self, dataset, lables):
        pairs = [(dataset[i], dataset[i + 1]) for i in range(0, len(dataset), 2)]
        labels = lables['glabel'].tolist()
        self.dataset_list = self.dataset_list + pairs
        self.labels_list = self.labels_list + labels


    def combine_dataset(self):
        self.transfer_dataset(self.dataset, self.labels)


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
            batched_graph1 = dgl.batch(list(tmp_graph1))
            batched_graph2 = dgl.batch(list(tmp_graph2))
            # print(batched_graph1.edata['w'].to(self.device))
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



# if __name__ == '__main__':
#     loaddata = LoadData()
#     loaddata.datasplit()

