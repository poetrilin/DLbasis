import torch
import torch.nn as nn
import timeit
from tqdm import tqdm
import os
from models import GCN, GCN_LP
from torch_geometric.data import Data
from typing import Callable, Tuple, Literal, Union
from torch import Tensor
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
# import numpy as np
from utils import plot_history, HistoryDict
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T


SEED = 42
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EARLY_STOPPING = 30
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LossFn = Callable[[Tensor, Tensor], Tensor]
Stage = Literal["train", "val", "test"]
Task = Literal["classification", "link_prediction"]


def accuracy(preds: Tensor, y: Tensor) -> float:
    return (preds == y).sum().item() / len(y)


def train_step(
    model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer, loss_fn: LossFn
) -> Tuple[float, float]:
    model.train()
    optimizer.zero_grad()
    mask = data.train_mask
    logits = model(data)[mask]
    preds = logits.argmax(dim=1)
    y = data.y[mask]
    loss = loss_fn(logits, y)
    # + L2 regularization to the first layer only
    # for name, params in model.state_dict().items():
    #     if name.startswith("conv1"):
    #         loss += 5e-4 * params.square().sum() / 2.0

    acc = accuracy(preds, y)
    loss.backward()
    optimizer.step()
    return loss.item(), acc


@torch.no_grad()
def eval_step(model: torch.nn.Module, data: Data, loss_fn: LossFn, stage: Stage) -> Tuple[float, float]:
    model.eval()
    mask = getattr(data, f"{stage}_mask")
    logits = model(data)[mask]
    preds = logits.argmax(dim=1)
    y = data.y[mask]
    loss = loss_fn(logits, y)
    # + L2 regularization to the first layer only
    # for name, params in model.state_dict().items():
    #     if name.startswith("conv1"):
    #         loss += 5e-4 * params.square().sum() / 2.0
    acc = accuracy(preds, y)
    return loss.item(), acc


def negative_sample(train_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
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


def train_link_step(model: nn.Module, data: Data, optimizer: torch.optim.Optimizer,
                    criterion: torch.nn.Module) -> Tuple[float, float]:
    model.train()
    optimizer.zero_grad()
    edge_label, edge_label_index = negative_sample(train_data=data)
    out = model(data.x, data.edge_index, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    roc_score = roc_auc_score(
        edge_label.cpu().detach().numpy(), out.cpu().detach().numpy())
    loss.backward()
    optimizer.step()
    return loss.item(), roc_score


@torch.no_grad()
def eval_link_step(model: nn.Module, data: Data, criterion: torch.nn.Module) -> Tuple[float, float]:
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    loss = criterion(out, data.edge_label)
    roc_score = roc_auc_score(data.edge_label.cpu(
    ).detach().numpy(), out.cpu().detach().numpy())
    return loss.item(), roc_score


def train(
    model: torch.nn.Module,
    data: Union[Tuple[Data, Data, Data], Data],
    *,
    optimizer: torch.optim.Optimizer,
    task: Task = "classification",
    max_epochs: int = 200,
    early_stopping: int = 30,
    print_interval: int = 20,
    verbose: bool = True,
) -> HistoryDict:
    history = {"loss": [], "val_loss": [], "metric": [], "val_metric": []}
    if task == "classification":
        metric_str = "acc"
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        metric_str = "roc_auc"
        loss_fn = torch.nn.BCEWithLogitsLoss()
        train_data, val_data, test_data = data
    start_time = timeit.default_timer()
    for epoch in range(max_epochs):
        if task == "classification":
            loss, metric = train_step(model, data, optimizer, loss_fn)
            val_loss, val_metric = eval_step(model, data, loss_fn, "val")
        else:
            loss, metric = train_link_step(
                model, train_data, optimizer, loss_fn)
            val_loss, val_metric = eval_link_step(model, val_data, loss_fn)
        history["loss"].append(loss)
        history["metric"].append(metric)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)
        # The official implementation in TensorFlow is a little different from what is described in the paper...
        if epoch > early_stopping and val_loss > np.mean(history["val_loss"][-(early_stopping + 1): -1]):
            if verbose:
                print("\nEarly stopping...")
            break

        if verbose and epoch % print_interval == 0:
            print(f"\nEpoch: {epoch}\n----------")
            print(f"Train loss: {loss:.4f} | Train {metric_str}: {metric:.4f}")
            print(
                f"  Val loss: {val_loss:.4f} |   Val {metric_str}: {val_metric:.4f}")

    end_time = timeit.default_timer()
    if task == "classification":
        test_loss, test_metric = eval_step(model, data, loss_fn, "test")
    else:
        test_loss, test_metric = eval_link_step(model, test_data, loss_fn)
    if verbose:
        print(f"\nTraining took {end_time - start_time:.2f} seconds")
    if verbose:
        print(f"\nEpoch: {epoch}\n----------")
        print(f"Train loss: {loss:.4f} | Train acc: {metric:.4f}")
        print(f"  Val loss: {val_loss:.4f} |   Val acc: {val_metric:.4f}")
        print(f" Test loss: {test_loss:.4f} |  Test acc: {test_metric:.4f}")

    return history


# task = "classification"
task = "link_prediction"
dataset_name = "Cora"

transform = None
cur_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(cur_dir, '../dataset/')


if task == "classification":
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
    ])
    dataset = Planetoid(root=dataset_dir, name="Cora", transform=transform)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    model = GCN(num_features, num_classes=num_classes).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = train(model, data, optimizer=optimizer, max_epochs=MAX_EPOCHS,
                    early_stopping=EARLY_STOPPING, task=task)
    title = "GCN"
else:
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    dataset = Planetoid(root=dataset_dir, name="Cora", transform=transform)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    model = GCN_LP(num_features, 128, 64).to(device)
    train_data, val_data, test_data = dataset[0]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = train(model, data=(train_data, val_data, test_data), optimizer=optimizer, max_epochs=MAX_EPOCHS,
                    early_stopping=EARLY_STOPPING, task=task)
    title = "GCN_LP"

plt.figure(figsize=(12, 4))
plot_history(history, title, task=task)
plt.show()
