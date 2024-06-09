import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Dropout
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = './model/'
model_dir = './model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(2, 32)
        self.dropout1 = Dropout(p=0.4)
        self.conv2 = GCNConv(32, 64)
        self.dropout2 = Dropout(p=0.4)
        self.conv3 = GCNConv(64, 32)
        self.dropout3 = Dropout(p=0.4)
        self.fc = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = global_mean_pool(x, batch)  # 使用全局平均池化
        x = self.fc(x)
        return x

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)  # 将数据移动到 GPU
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1, 1))  # 确保目标形状匹配
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    error_sum = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)  # 将数据移动到 GPU
            out = model(data)
            error = torch.abs(out - data.y.view(-1, 1))
            error_sum += error.sum().item()
    error_avg = error_sum / len(loader.dataset)
    return error_avg

files = ['train_set_500.pt', 'train_set_1000.pt', 'train_set_1500.pt', 'train_set_2000.pt']
full_processed_data = []
for file_name in files:
    dataset = torch.load(file_name)
    processed_data = []
    for data in dataset:
        data_batch = {}
        x = torch.cat([data['node_type'], data['num_inverted_predecessors']], dim=1)
        x = x.reshape(2, -1).T
        x = torch.tensor(x, dtype=torch.float32)
        edge_index = data['edge_index']
        new_data = Data(x=x, edge_index=edge_index, y=data['target'])
        processed_data.append(new_data)
    full_processed_data.extend(processed_data)
train_data, test_data = train_test_split(full_processed_data, test_size=0.2, shuffle=True, random_state=42)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Initialize model, optimizer, and loss function
model = GNNModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 调整学习率
criterion = nn.MSELoss()

# Training loop
losses = []
for epoch in range(400):  # 增加训练轮数
    loss = train(model, train_loader, optimizer, criterion)
    print(f'Epoch {epoch+1}, Loss: {loss}')
    losses.append(loss)

state_dict = model.state_dict()
torch.save(state_dict, os.path.join(model_dir, 'task1_model_weights_400.pth'))

# Test the model
error = test(model, test_loader)
print(f'Test Error: {error}')

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(model_dir, "task1_training_loss_400.png"), format='png')

