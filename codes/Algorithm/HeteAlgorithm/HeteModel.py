# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi
@File    ：HeteModel.py
@Author  ：honywen
@Date    ：2023/6/19 04:28
@Software: PyCharm
"""


import dgl
import torch
import torch.nn as nn
import loaddata_heter
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GatedGraphConv, GATConv, HeteroGraphConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class HGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, num_classes):
        super(HGCN, self).__init__()
        # self.conv1 = HeteroGraphConv({rel: GraphConv(in_feats, hid_feats) for rel in rel_names}, aggregate='sum')
        self.conv1 = HeteroGraphConv({rel: GraphConv(in_feats, hid_feats) for rel in rel_names}, aggregate='sum')
        self.gatconv = HeteroGraphConv({rel: GATConv(hid_feats, hid_feats, num_heads=3) for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({rel: GraphConv(hid_feats, out_feats) for rel in rel_names}, aggregate='sum')
        self.classify = nn.Linear(out_feats, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, graph, inputs):
        # inputs是节点的特征
        h = self.conv1(graph, inputs)
        h = {k: F.normalize(v) for k, v in h.items()}
        h = self.gatconv(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, 0.3) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: torch.sigmoid(v) for k, v in h.items()}
        with graph.local_scope():
            graph.ndata['n'] = h
            # 通过平均读出值来计算单图的表征
            hg = 0
            for ntype in graph.ntypes:
                try:
                    hg = hg + dgl.mean_nodes(graph, 'n', ntype=ntype)
                except:
                    pass
            hg = self.classify(hg)
            hg = torch.sigmoid(hg)
            hg = torch.mean(hg, dim=2)
            return hg


class SiameseGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_feats, rel_names, n_classes):
        super(SiameseGCN, self).__init__()
        self.hetegcn = HGCN(in_dim, hidden_dim, out_feats, rel_names, n_classes)

    def forward(self, g1, g2):
        # (32,1,2)
        h1 = self.hetegcn(g1, g1.ndata['hy'])
        h2 = self.hetegcn(g2, g2.ndata['hy'])
        # dist = F.cosine_similarity(h1, h2)
        dist = F.pairwise_distance(h1, h2, p=2)
        # dist = F.softmax(dist, dim=1)
        return dist


class TrainTest:
    def __init__(self, train_dataloader, validation_dataloader, test_dataloader):
        self.dim_nfeats = 100
        self.hidden1_feats = 200
        self.hidden2_feats = 250
        self.gclasses = 2
        self.batchepoch = 50000
        self.num_correct = 0
        self.num_tests = 0
        self.min_loss = 1000
        self.loss_list = list()
        self.acc_list = list()
        self.valid_loss_list = list()
        self.valid_acc_list = list()
        self.test_real_label = list()
        self.test_pred_label = list()
        self.train_dataloader = list()
        self.test_dataloader = list()
        self.pred_prob_list = list()
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.labels = [0, 1]
        self.etypes = ['1', '2']
        self.drawlabels = ['0', '1']
        self.classes = ['normal', 'abnormal']
        self.max_patience = 10
        self.patience = 0
        self.loss_function = nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseGCN(self.dim_nfeats, self.hidden1_feats, self.hidden2_feats, self.etypes, self.gclasses).to(self.device)
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=0.001)
        # self.contrastiveloss = ContrastiveLoss()

    def verifymodel(self):
        self.model.eval()
        correct = 0.
        loss = 0.
        iter = 0
        for iter, (batched_graph, labels) in enumerate(self.validation_dataloader):
            out = self.model(batched_graph[0].to(self.device),batched_graph[1].to(self.device))
            labels_tmp = labels.view(-1, 1)
            out = torch.clamp(out, 0, 1)
            loss += self.loss_function(out, labels_tmp).detach().item()
            probs_Y = (out >= 0.5).float()
            test_Y = labels.clone().detach().float().view(-1, 1).to(self.device)
            acc_ = (test_Y == probs_Y).sum().item() / len(test_Y) * 100
            correct += acc_
        correct /= (iter + 1)
        loss /= (iter + 1)
        return correct, loss

    def trainmodel(self):
        for epoch in range(self.batchepoch):
            train_preds = []
            train_trues = []
            for batched_graph, labels in self.train_dataloader:
                pred = self.model(batched_graph[0].to(self.device), batched_graph[1].to(self.device))
                # print(pred)
                labels_tmp = labels.view(-1, 1)
                pred = torch.clamp(pred, 0, 1)
                loss = self.loss_function(pred, labels_tmp)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # train_outputs = pred.argmax(1)
                # print(train_outputs)
                train_outputs = (pred >= 0.5).float()
                train_outputs = torch.squeeze(train_outputs).long()
                labels = labels.cpu()
                train_outputs = train_outputs.cpu()
                for pred_label in train_outputs:
                    train_preds.append(pred_label)
                for real_label in labels:
                    train_trues.append(real_label)
            sklearn_accuracy = accuracy_score(train_trues, train_preds)
            self.acc_list.append(sklearn_accuracy)
            self.loss_list.append(loss.item())
            print("Epoch {:05d}  | Loss {:.4f} |  Acc {:.4f}".format(epoch, loss.item(), sklearn_accuracy))
            val_acc, val_loss = self.verifymodel()
            self.valid_loss_list.append(val_loss)
            self.valid_acc_list.append(val_acc)
            print("Validation loss:{}\taccuracy:{}".format(val_loss, val_acc))
            if val_loss < self.min_loss:
                torch.save(self.model.state_dict(), 'latest_GAT_GGCN_cd_binutils_optimization2' + '.pkl')
                print("Model saved at epoch{}".format(epoch))
                self.min_loss = val_loss
                self.patience = 0
            else:
                self.patience += 1
            # if self.patience > self.max_patience:
            #     print('patience out!')
            #     break

    def testmodel(self):
        self.model.load_state_dict(torch.load('latest_GAT_GGCN_cd_binutils_optimization2.pkl'))
        for batched_graph, labels in self.test_dataloader:
            pred = self.model(batched_graph[0].to(self.device), batched_graph[1].to(self.device))
            self.pred_prob_list = self.pred_prob_list + pred.cpu().tolist()
            train_outputs = (pred >= 0.5).float()
            train_outputs = torch.squeeze(train_outputs).long()
            pred_list = train_outputs.cpu().numpy().tolist()
            real_list = labels.cpu().numpy().tolist()
            for pred_label in pred_list:
                self.test_pred_label.append(pred_label)
            for real_label in real_list:
                self.test_real_label.append(real_label)
            self.num_correct += (pred.argmax(1) == labels).sum().item()
            self.num_tests += len(labels)

    def model_evaluation(self):
        sklearn_accuracy = accuracy_score(self.test_real_label, self.test_pred_label)
        sklearn_precision = precision_score(self.test_real_label, self.test_pred_label, average='weighted')
        sklearn_recall = recall_score(self.test_real_label, self.test_pred_label, average='weighted')
        sklearn_f1 = f1_score(self.test_real_label, self.test_pred_label, average='weighted')
        print(sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1)
        print('Test precision:', sklearn_precision)
        print('Test recall:', sklearn_recall)
        print('Test f1:', sklearn_f1)
        print('Test accuracy:', self.num_correct / self.num_tests)

    def get_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.test_real_label, self.test_pred_label)
        return conf_matrix


if __name__ == '__main__':
    # 获取数据集
    loaddata = loaddata_heter.LoadData()
    # 划分数据集
    train_dataloader, validation_dataloader, test_dataloader = loaddata.datasplit()
    # 加载数据集
    modeltraintest = TrainTest(train_dataloader, validation_dataloader, test_dataloader)
    # 模型训练
    modeltraintest.trainmodel()
    # # 模型测试
    modeltraintest.testmodel()
    # # 模型评估
    modeltraintest.model_evaluation()
    # # 混淆矩阵
    # conf_matrix = modeltraintest.get_confusion_matrix()
    # modeltraintest.draw_trainaccloss()
    # modeltraintest.draw_validaccloss()
    # modeltraintest.draw_roc()
    # modeltraintest.plot_confusion_matrix(conf_matrix)

