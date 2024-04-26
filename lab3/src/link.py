import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import copy
from models import GCN_LP
import os
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)


def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        model.train()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


def negative_sample(train_data):
    # 从训练集中采样与正边相同数量的负边
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    # print(neg_edge_index.size(1))
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    return edge_label, edge_label_index


def train(train_data, val_data, test_data, model=None):
    if model is None:
        model = GCN_LP(dataset.num_features, 128, 64).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    min_epochs = 10
    best_val_auc = 0
    final_test_auc = 0
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        edge_label, edge_label_index = negative_sample(train_data=train_data)
        out = model(train_data.x, train_data.edge_index,
                    edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        # validation
        val_auc = test(model, val_data)
        test_auc = test(model, test_data)
        if epoch + 1 > min_epochs and val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc

        print('epoch {:03d} train_loss {:.8f} val_auc {:.4f} test_auc {:.4f}'
              .format(epoch, loss.item(), val_auc, test_auc))

    return final_test_auc


cur_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(cur_dir, '../dataset/')
save_model_dir = os.path.join(cur_dir, '../model/')
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)
# save_model_path = os.path.join(save_model_dir, 'gcn_lp.pth')


transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])
dataset = Planetoid(root=dataset_dir, name="Cora", transform=transform)
train_data, val_data, test_data = dataset[0]

num_features = dataset.num_features

model = GCN_LP(dataset.num_features, 128, 64).to(device)

train(train_data, val_data, test_data, model=model)
