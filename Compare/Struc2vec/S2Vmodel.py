import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from Compare.Struc2vec.splitdata import LoadData
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Define a structure2vec layer using PyTorch Geometric's MessagePassing
class S2VLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(S2VLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j denotes the features of neighboring nodes as determined by edge_index
        return self.lin(x_j)

    def update(self, aggr_out, x):
        # aggr_out denotes the aggregated messages from the neighbors
        new_embedding = torch.cat([x, aggr_out], dim=1)
        return F.relu(self.lin_update(new_embedding))

# Define the structure2vec network
class Structure2VecNetwork(nn.Module):
    def __init__(self, feature_dim, embed_dim, hidden_dim, depth):
        super(Structure2VecNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(S2VLayer(feature_dim, hidden_dim))
        for i in range(depth - 1):
            self.layers.append(S2VLayer(hidden_dim, hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer(x, edge_index)
        x = torch.tanh(self.out_proj(x))

        # Global pooling to ensure same size embeddings
        x_pooled = global_mean_pool(x, batch)
        return x_pooled

# Define the Siamese network architecture
class SiameseNetwork(nn.Module):
    def __init__(self, feature_dim, embed_dim, hidden_dim, depth):
        super(SiameseNetwork, self).__init__()
        self.s2v = Structure2VecNetwork(feature_dim, embed_dim, hidden_dim, depth)

    def forward(self, data1, data2):
        embed1 = self.s2v(data1)
        print(embed1.shape)
        embed2 = self.s2v(data2)
        print(embed2.shape)
        # Ensure same size embeddings before cosine similarity
        cos_similarity = F.cosine_similarity(embed1, embed2, dim=1)
        return cos_similarity


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output, label):
        # For simplicity, let's assume label 1 means similar and label 0 means dissimilar
        margin = 1
        loss = label * (1 - output) ** 2 + (1 - label) * torch.clamp(output - margin, min=0) ** 2
        return loss.mean()


class SiameseTrainer:
    def __init__(self, feature_dim, embed_dim, hidden_dim, depth, lr=0.001):
        self.model = SiameseNetwork(feature_dim, embed_dim, hidden_dim, depth)
        self.criterion = ContrastiveLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            total_loss = 0
            all_labels = []
            all_predictions = []

            for graph_pairs, labels in data_loader:
                print(labels)
                for (data1, data2), label in zip(graph_pairs, labels):
                    print(data1)
                    print(label)
                    # 转移到设备
                    data1 = graph_pairs[0].to(self.device)
                    data2 = graph_pairs[1].to(self.device)
                    label = labels
                    print(data1)

                    # 前向传播
                    self.optimizer.zero_grad()
                    similarity_score = self.model(data1, data2)
                    print(similarity_score)
                    predictions = (similarity_score > 0.5).float()
                    print(predictions)
                    print(label)
                    # 计算损失并反向传播
                    loss = self.criterion(similarity_score, label)
                    loss.backward()
                    self.optimizer.step()

                    # 累计损失和预测
                    total_loss += loss.item()
                    all_labels.append(label.item())
                    all_predictions.append(predictions.item())

            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions)
            recall = recall_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions)

            print(f'Epoch {epoch}, Loss: {total_loss:.4f}, '
                  f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
                  f'Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    def test(self, data_loader):
        self.model.eval()  # Set the model to evaluation mode
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in data_loader:
                for ((data1, data2), label) in batch:
                    data1 = data1.to(self.device)
                    data2 = data2.to(self.device)
                    label = torch.tensor(label, dtype=torch.float32).to(self.device)

                    similarity_score = self.model(data1, data2)
                    predictions = (similarity_score > 0.5).float()

                    all_labels.extend(label.tolist())
                    all_predictions.extend(predictions.tolist())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        return accuracy, precision, recall, f1


if __name__ == '__main__':
    # Parameters
    feature_dim = 8
    embed_dim = 10
    hidden_dim = 20
    depth = 3
    batch_size = 2
    epochs = 10

    loaddata = LoadData()
    train_loader, val_loader, test_loader = loaddata.split_data()
    trainer = SiameseTrainer(feature_dim, embed_dim, hidden_dim, depth)
    # Training
    trainer.train(train_loader, epochs)
    # Validation
    val_accuracy, val_precision, val_recall, val_f1 = trainer.test(val_loader)
    print(f'Validation - Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}')
    # Testing
    test_accuracy, test_precision, test_recall, test_f1 = trainer.test(test_loader)
    print(f'Test - Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}')


