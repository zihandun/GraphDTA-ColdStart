"""
Zihan Dun
2025/11/17
"""

import os
from random import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed Set] seed={seed}")

def train(model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        # for data in tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch"):
        for data in tqdm(loader):
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1,1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

datasets = ['davis', 'kiba']  # [, 'kiba']
modeling = GINConvNet  # GATNet, GAT_GCN, GCNNet
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# 设置和论文一致了
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
NUM_EPOCHS = 100
for i in range(1001, 1001 + 1):
    # set_seed(0 + i - 1)
    for dataset in datasets:
        print(f'\n========== Running on {model_st}_{dataset} , Seed:{0 + i - 1}==========')

        train_file = f'data/processed/{dataset}_train.pt'
        val_file = f'data/processed/{dataset}_val.pt'
        test_split = 'test'  # 'test1'  'test2'
        test_file = f'data/processed/{dataset}_{test_split}.pt'

        for f in [train_file, val_file, test_file]:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"File not found: {f}. Run create_data.py firstly")

        train_data = TestbedDataset(root='data', dataset=f'{dataset}_train')
        val_data = TestbedDataset(root='data', dataset=f'{dataset}_val')
        test_data = TestbedDataset(root='data', dataset=f'{dataset}_{test_split}')
        test1_data = TestbedDataset(root='data', dataset=f'{dataset}_test1')
        test2_data = TestbedDataset(root='data', dataset=f'{dataset}_test2')

        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test1_loader = DataLoader(test1_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test2_loader = DataLoader(test2_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        list_test = [test_loader, test1_loader, test2_loader]
        print(f"\n========== Data {dataset}==========")
        print(f"len_train_set: {len(train_data)}")
        print(f"len_val_set: {len(val_data)}")
        print(f"len_unseenpair_test_set: {len(test_data)}")
        print(f"len_unseendrug_test1_set: {len(test1_data)}")
        print(f"len_unseentarget_test2_set: {len(test2_data)}")
        """
        KIBA 229 2111 118,254
        Davis 442 68 30,056
        """
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_mse = float('inf')
        best_epoch = -1
        model_file_name = f'model_{model_st}_{dataset}.model'
        result_file_name = f'result_{model_st}_{dataset}.csv'
        """
        """
        for epoch in range(1, NUM_EPOCHS+1):
            avg_loss = train(model, device, train_loader, optimizer, loss_fn, epoch)
            print(f"Epoch {epoch} finished. Average training loss: {avg_loss:.6f}")

            # TODO: Ci方法是O(n^2)的，改为np方式
            G_val, P_val = predicting(model, device, val_loader)
            val_metrics = [mae(G_val, P_val), mse(G_val, P_val), ci(G_val, P_val), r2(G_val, P_val)]
            print(f"Validation metrics: MAE={val_metrics[0]:.4f}, MSE={val_metrics[1]:.4f}, "
                  f"CI={val_metrics[2]:.4f}")

            if val_metrics[1] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, val_metrics)))
                best_epoch = epoch
                best_mse = val_metrics[1]
                best_ci = val_metrics[-2]
                print(f'Validation MSE improved at epoch {best_epoch}; model saved! Best MSE: {best_mse:.6f}, Best CI: {best_ci:.4f}')
            else:
                print(f'No improvement at epoch {epoch}. Current best MSE: {best_mse:.6f}, Best CI: {best_ci:.4f}')
       
        model.load_state_dict(torch.load(model_file_name))
        for i in range(0, 3):
            test_loader = list_test[i]
            G_test, P_test = predicting(model, device, test_loader)
            test_metrics = [mae(G_test, P_test), mse(G_test, P_test), ci(G_test, P_test), r2(G_test, P_test)]
            print(f'Test metrics for {dataset} (test {i}): MAE={test_metrics[0]:.4f}, MSE={test_metrics[1]:.4f}, '
                f'CI={test_metrics[2]:.4f}, R=square={test_metrics[3]:.4f}')
