# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi 
@File    ：model.py
@Author  ：honywen
@Date    ：2023/7/16 20:31 
@Software: PyCharm
"""


import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from Compare.GNNAcfg.loaddata import LoadData
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



class GCN(nn.Module):
    def __init__(self, in_feats, h2_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h2_feats)
        self.classify = nn.Linear(h2_feats, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.normalize(h)
        h = torch.sigmoid(h)
        h = F.relu(h)
        h = F.dropout(h, 0.3)
        with g.local_scope():
            g.ndata['h'] = h.to(self.device)
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)

class SiameseGCN(nn.Module):
    def __init__(self, in_feats, h2_feats, num_classes):
        super(SiameseGCN, self).__init__()
        self.gcn = GCN(in_feats, h2_feats, num_classes)

    def forward(self, g1, g2, in_feat1, in_feat2):
        h1 = self.gcn(g1, in_feat1)
        h2 = self.gcn(g2, in_feat2)
        dist = F.pairwise_distance(h1, h2)
        return dist


class TrainTest:
    def __init__(self, train_dataloader, validation_dataloader, test_dataloader):
        self.dim_nfeats = 8
        self.hidden2_feats = 8
        self.gclasses = 8
        self.batchepoch = 10
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
        self.max_patience = 10
        self.patience = 0
        self.model = SiameseGCN(self.dim_nfeats,self.hidden2_feats,self.gclasses)
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=0.0001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def verifymodel(self):
        self.model.eval()
        correct = 0.
        loss = 0.
        for iter, (batched_graph, labels) in enumerate(self.validation_dataloader):
            nfeats1 = batched_graph[0].ndata['x']
            nfeats2 = batched_graph[1].ndata['x']
            out = self.model(batched_graph[0], batched_graph[1], nfeats1, nfeats2)
            loss += F.cross_entropy(out, labels).detach().item()
            # probs_Y = torch.softmax(out, 0)
            test_Y = labels.clone().detach().float().view(-1, 1).to(self.device)
            argmax_Y = (out >= 0.5).float().view(-1, 1).to(self.device)
            acc_ = (test_Y == argmax_Y).sum().item() / len(test_Y) * 100
            correct += acc_
        correct /= (iter + 1)
        loss /= (iter + 1)
        return correct, loss

    def trainmodel(self):
        for epoch in range(self.batchepoch):
            train_preds = []
            train_trues = []
            for batched_graph, labels in self.train_dataloader:
                nfeats1 = batched_graph[0].ndata['x']
                nfeats2 = batched_graph[1].ndata['x']
                pred = self.model(batched_graph[0], batched_graph[1], nfeats1, nfeats2)
                loss = F.cross_entropy(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_outputs = (pred >= 0.5).float()
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
                torch.save(self.model.state_dict(), 'latest_GAT_GGCN_' + '.pkl')
                print("Model saved at epoch{}".format(epoch))
                self.min_loss = val_loss
                self.patience = 0
            else:
                self.patience += 1
            # if self.patience > self.max_patience:
            #     print('patience out!')
            #     break

    def testmodel(self):
        for batched_graph, labels in self.test_dataloader:
            nfeats1 = batched_graph[0].ndata['x']
            nfeats2 = batched_graph[1].ndata['x']
            pred = self.model(batched_graph[0], batched_graph[1], nfeats1, nfeats2)
            self.pred_prob_list = self.pred_prob_list + pred.cpu().tolist()
            pred_list = (pred >= 0.5).cpu().numpy().tolist()
            real_list = labels.cpu().numpy().tolist()
            for pred_label in pred_list:
                self.test_pred_label.append(pred_label)
            for real_label in real_list:
                self.test_real_label.append(real_label)
            self.num_correct += ((pred >= 0.5) == labels).sum().item()
            self.num_tests += len(labels)

    def model_evaluation(self):
        sklearn_accuracy = accuracy_score(self.test_real_label, self.test_pred_label)
        sklearn_precision = precision_score(self.test_real_label, self.test_pred_label, average='weighted')
        sklearn_recall = recall_score(self.test_real_label, self.test_pred_label, average='weighted')
        sklearn_f1 = f1_score(self.test_real_label, self.test_pred_label, average='weighted')
        print('Test precision:', sklearn_precision)
        print('Test recall:', sklearn_recall)
        print('Test f1:', sklearn_f1)
        print('Test accuracy:', self.num_correct / self.num_tests)


if __name__ == '__main__':
    loaddata = LoadData()
    train_dataloader, validation_dataloader, test_dataloader = loaddata.datasplit()
    modeltraintest = TrainTest(train_dataloader, validation_dataloader, test_dataloader)
    modeltraintest.trainmodel()
    modeltraintest.testmodel()
    modeltraintest.model_evaluation()