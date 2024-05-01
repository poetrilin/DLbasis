from typing import List, Optional, TypedDict, Literal
import matplotlib.pyplot as plt
from torch_geometric.data import Data

Task = Literal["classification", "link_prediction"]


def re_split(data: Data, train_rate: float, val_rate: float) -> Data:
    assert train_rate+val_rate < 1
    test_rate = 1-train_rate-val_rate
    data.train_mask.fill_(False)
    data.val_mask.fill_(False)
    data.test_mask.fill_(False)
    data.train_mask[:int(data.num_nodes*train_rate)] = True
    data.val_mask[int(data.num_nodes*train_rate)
                      :int(data.num_nodes*(train_rate+val_rate))] = True
    data.test_mask[int(data.num_nodes*(train_rate+val_rate)):] = True
    return data


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
