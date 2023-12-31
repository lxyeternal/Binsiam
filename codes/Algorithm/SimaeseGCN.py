# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi 
@File    ：SimaeseGCN.py
@Author  ：honywen
@Date    ：2023/5/29 00:06 
@Software: PyCharm
"""


import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from codes.Algorithm.loaddata import LoadData
from dgl.nn.pytorch import GraphConv, GatedGraphConv, GATConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class GGANN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, num_classes, n_steps, n_etypes):
        super(GGANN, self).__init__()
        self.gatedconv1 = GraphConv(in_feats, h1_feats)
        self.gatconv = GATConv(h1_feats, h2_feats, num_heads=2)
        self.gatedconv2 = GraphConv(h2_feats, h2_feats)
        self.classify = nn.Linear(h2_feats, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, g, in_feat):
        h = self.gatedconv1(g, in_feat)
        h = F.normalize(h)
        h = torch.sigmoid(h)
        h = self.gatconv(g, h)
        h = F.relu(h)
        h = F.dropout(h, 0.3)
        h = self.gatedconv2(g, h)
        with g.local_scope():
            g.ndata['h'] = h.to(self.device)
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)

class SiameseGCN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, num_classes, n_steps, n_etypes):
        super(SiameseGCN, self).__init__()
        self.gcn = GGANN(in_feats, h1_feats, h2_feats, num_classes, n_steps, n_etypes)

    def forward(self, g1, g2, in_feat1, in_feat2, etype1=None, etype2=None):
        h1 = self.gcn(g1, in_feat1)
        h2 = self.gcn(g2, in_feat2)
        dist = F.pairwise_distance(h1, h2, 2)
        data_normalized = F.softmax(dist, dim=1)
        return data_normalized


class TrainTest:
    def __init__(self, train_dataloader, validation_dataloader, test_dataloader):
        self.dim_nfeats = 100
        self.hidden1_feats = 200
        self.hidden2_feats = 128
        self.gclasses = 2
        self.n_steps = 10
        self.n_etypes = 3
        self.batchepoch = 100
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
        self.drawlabels = ['0', '1']
        self.classes = ['normal','abnormal']
        self.max_patience = 10
        self.patience = 0
        self.model = SiameseGCN(self.dim_nfeats,self.hidden1_feats,self.hidden2_feats,self.gclasses,self.n_steps,self.n_etypes)
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=0.0001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def verifymodel(self):
        self.model.eval()
        correct = 0.
        loss = 0.
        for iter, (batched_graph, labels) in enumerate(self.validation_dataloader):
            nfeats1 = batched_graph[0].ndata['x']
            efeats1 = batched_graph[0].edata['w']
            nfeats2 = batched_graph[1].ndata['x']
            efeats2 = batched_graph[1].edata['w']
            out = self.model(batched_graph[0], batched_graph[1], nfeats1, nfeats2, efeats1, efeats2)
            loss += F.cross_entropy(out, labels.long()).detach().item()
            probs_Y = torch.softmax(out, 1)
            test_Y = labels.clone().detach().float().view(-1, 1).to(self.device)
            argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1).to(self.device)
            acc_ = (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100
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
                efeats1 = batched_graph[0].edata['w']
                nfeats2 = batched_graph[1].ndata['x']
                efeats2 = batched_graph[1].edata['w']
                pred = self.model(batched_graph[0], batched_graph[1], nfeats1, nfeats2)
                loss = F.cross_entropy(pred, labels.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_outputs = pred.argmax(1)
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
            efeats1 = batched_graph[0].edata['w']
            nfeats2 = batched_graph[1].ndata['x']
            efeats2 = batched_graph[1].edata['w']
            pred = self.model(batched_graph[0], batched_graph[1], nfeats1, nfeats2, efeats1, efeats2)
            self.pred_prob_list = self.pred_prob_list + pred.cpu().tolist()
            pred_list = pred.argmax(1).cpu().numpy().tolist()
            real_list = labels.cpu().numpy().tolist()
            for pred_label in pred_list:
                self.test_pred_label.append(pred_label)
            for real_label in real_list:
                self.test_real_label.append(real_label)
            self.num_correct += (pred.argmax(1) == labels).sum().item()
            self.num_tests += len(labels)


if __name__ == '__main__':
    loaddata = LoadData()
    train_dataloader, validation_dataloader, test_dataloader = loaddata.datasplit()
    modeltraintest = TrainTest(train_dataloader,validation_dataloader,test_dataloader)
    modeltraintest.trainmodel()