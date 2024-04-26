from typing import TypedDict, Literal, Tuple, List, Optional
import torch_geometric.transforms as T
import torch
SEED = 42
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EARLY_STOPPING = 10

ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
device = "cuda" if torch.cuda.is_available() else "cpu"

metric_dict = {
    "clssification": "accuracy",
    "link_prediction": "roc_auc_score"
}

transfrom_dict = {
    "classification": T.compose([
        T.NormalizeFeatures(),
        T.AddSelfLoops(),
        T.ToDevice(device),
    ]),
    "link_prediction": T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
}
