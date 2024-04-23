import torch
import torch.nn as nn
from torch import optim
import timeit
from tqdm import tqdm
from utils import load_data
# from model import MyNet3, Alexnet
import os
import torch.nn.functional as F


SEED = 42
torch.manual_seed(SEED)
# Hyper Parameters


EPOCHS = 12
BATCH_SIZE = 64
# Model Parameters
IN_CHANNELS = 3
DROPOUT = 0.1
NUM_CLASSES = 7

LEARNING_RATE = 5e-4
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
DATA_DIR = "./dataset/"
SAVE_DIR = "./model/"


def train(model, data_dir=DATA_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
          adam_weight_decay=ADAM_WEIGHT_DECAY, adam_betas=ADAM_BETAS, save_dir=SAVE_DIR):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    train_dataloader, val_dataloader, test_dataloader = load_data(
        data_dir=data_dir, batch_size=batch_size)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), betas=adam_betas,
                           lr=learning_rate, weight_decay=adam_weight_decay)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    start = timeit.default_timer()
    val_losses = []

    for epoch in tqdm(range(epochs), position=0, leave=True):
        model.train()
        train_labels = []
        train_preds = []
        train_running_loss = 0
        for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img, label = img_label
            img = img.float().to(device)
            label = label.type(torch.uint8).to(device)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim=1)

            train_labels.extend(label.cpu().detach())
            train_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(y_pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
        train_loss = train_running_loss / (idx + 1)

        # validation
        model.eval()
        val_labels = []
        val_preds = []
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_label in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img, label = img_label
                img = img.float().to(device)
                label = label.type(torch.uint8).to(device)
                y_pred = model(img)
                y_pred_label = torch.argmax(y_pred, dim=1)
                val_labels.extend(label.cpu().detach())
                val_preds.extend(y_pred_label.cpu().detach())
                loss = criterion(y_pred, label)
                val_running_loss += loss.item()
        val_loss = val_running_loss / (idx + 1)
        val_losses.append(val_loss)

        print("-" * 30)
        print(f"Train Loss Epoch {epoch+1} : {train_loss:.4f}")
        print(f"Val Loss Epoch {epoch+1} : {val_loss:.4f}")
        print(
            f"Train Accuracy EPOCH {epoch + 1}: {sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
        print(
            f"Val Accuracy EPOCH {epoch + 1}: {sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
        print("-" * 30)

    stop = timeit.default_timer()
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    model_path = save_dir + "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training Time:{stop - start:.2f}s")
    return val_losses
