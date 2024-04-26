import torch
import torch.nn as nn
import timeit
from tqdm import tqdm
import os
from models import GCN
from torch_geometric.data import Data
from typing import Callable, Tuple, Literal, List, Optional, TypedDict
from torch import Tensor
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
# import numpy as np

SEED = 42
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EARLY_STOPPING = 10
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)

torch.manual_seed(SEED)
# os.environ[“KMP_DUPLICATE_LIB_OK”]=“TRUE”
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LossFn = Callable[[Tensor, Tensor], Tensor]
Stage = Literal["train", "val", "test"]
Task = Literal["classification", "link_prediction"]


class HistoryDict(TypedDict):
    loss: List[float]
    metric: List[float]
    val_loss: List[float]
    val_metric: List[float]


def plot_history(history: HistoryDict, title: str, font_size: Optional[int] = 14,
                 task: Task = "classification") -> None:
    plt.suptitle(title, fontsize=font_size)
    ax1 = plt.subplot(121)
    ax1.set_title("Loss")
    ax1.plot(history["loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    ax1.legend()

    ax2 = plt.subplot(122)
    if task == "classification":
        ax2.set_title("Accuracy")
    else:
        ax2.set_title("roc_auc_score")
    ax2.plot(history["metric"], label="train")
    ax2.plot(history["val_metric"], label="val")
    plt.xlabel("Epoch")
    ax2.legend()


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


def train(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn = torch.nn.CrossEntropyLoss(),
    max_epochs: int = 200,
    early_stopping: int = 10,
    print_interval: int = 20,
    verbose: bool = True,
) -> HistoryDict:
    history = {"loss": [], "val_loss": [], "metric": [], "val_metric": []}
    start_time = timeit.default_timer()
    for epoch in range(max_epochs):
        loss, acc = train_step(model, data, optimizer, loss_fn)
        val_loss, val_acc = eval_step(model, data, loss_fn, "val")
        history["loss"].append(loss)
        history["metric"].append(acc)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_acc)
        # The official implementation in TensorFlow is a little different from what is described in the paper...
        # if epoch > early_stopping and val_loss > np.mean(history["val_loss"][-(early_stopping + 1): -1]):
        #     if verbose:
        #         print("\nEarly stopping...")
        #     break

        if verbose and epoch % print_interval == 0:
            print(f"\nEpoch: {epoch}\n----------")
            print(f"Train loss: {loss:.4f} | Train acc: {acc:.4f}")
            print(f"  Val loss: {val_loss:.4f} |   Val acc: {val_acc:.4f}")

    end_time = timeit.default_timer()
    if verbose:
        print(f"\nTraining took {end_time - start_time:.2f} seconds")
    test_loss, test_acc = eval_step(model, data, loss_fn, "test")
    if verbose:
        print(f"\nEpoch: {epoch}\n----------")
        print(f"Train loss: {loss:.4f} | Train acc: {acc:.4f}")
        print(f"  Val loss: {val_loss:.4f} |   Val acc: {val_acc:.4f}")
        print(f" Test loss: {test_loss:.4f} |  Test acc: {test_acc:.4f}")

    return history


task = "classification"
dataset_name = "Cora"

transform = None
cur_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(cur_dir, '../dataset/')
dataset = Planetoid(root=dataset_dir, name="Cora", transform=transform)
num_features = dataset.num_features
num_classes = dataset.num_classes


model = GCN(num_features, num_classes=num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

history = train(model, data, optimizer, max_epochs=MAX_EPOCHS,
                early_stopping=EARLY_STOPPING)


plt.figure(figsize=(12, 4))
plot_history(history, "GCN")
plt.show()
