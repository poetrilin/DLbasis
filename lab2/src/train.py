import torch
import torch.nn as nn
from torch import optim
import timeit
from tqdm import tqdm
from util import load_data
from model import LeNet5
import plotly.express as px

SEED = 24
torch.manual_seed(SEED)
# Hyper Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

EPOCHS = 20
BATCH_SIZE = 8
# Model Parameters
IN_CHANNELS = 3
DROPOUT = 0.001
NUM_CLASSES = 10

LEARNING_RATE = 1e-3
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
data_dir = "../dataset/"

train_dataloader, val_dataloader, test_dataloader = load_data(
    data_dir=data_dir, batch_size=BATCH_SIZE)

model = LeNet5().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS,
                       lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)

start = timeit.default_timer()
val_losses = []

for epoch in tqdm(range(EPOCHS), position=0, leave=True):
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
print(f"Training Time:{stop - start:.2f}s")

x_data = range(EPOCHS)
fig = px.scatter(x=x_data,
                 y=val_losses,
                 color=x_data,  # 颜色设置
                 text=x_data  # 显示内容
                 )
fig.update_traces(textposition="top center")  # 显示位置：顶部居中

fig.show()
