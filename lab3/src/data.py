import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.transforms import NormalizeFeatures, AddSelfLoops
from torch_geometric.utils import negative_sampling
import os
from tqdm import tqdm
from typing import Literal, Tuple

Task = Literal["classification", "link_prediction"]
Dataset = Literal["cora", "citeseer"]


def load_data(dataset_dir: str, *, name: Dataset = "cora",
              transform: T = None, task: Task = "classification") -> Data:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if task == "classification":
        transform = T.Compose([
            NormalizeFeatures(),
            # AddSelfLoops(),
            T.ToDevice(device),
        ])
    elif task == "link_prediction":
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                              add_negative_train_samples=False),
        ])
    dataset = Planetoid(dataset_dir, name, transform=transform)
   # 应用变换
    if transform is not None:
        dataset.transform = transform
    return dataset


# transform = torch_geometric.transforms.Compose
# ([NormalizeFeatures(), AddSelfLoops()])

# cur_dir = os.path.dirname(__file__)
# dataset_dir = os.path.join(cur_dir, '../dataset/')

# data = load_data(dataset_dir, transform=transform)
# print(data)

def negative_sample(data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    # 从训练集中采样与正边相同数量的负边
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1), method='sparse')
    # print(neg_edge_index.size(1))

    edge_label_index = torch.cat(
        [data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        data.edge_label,
        data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    return edge_label, edge_label_index
