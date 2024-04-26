from typing import List, Optional, TypedDict, Literal
import matplotlib.pyplot as plt

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
