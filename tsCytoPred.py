import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd 
import os
import numpy as np
import sys
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data, idx_data):
        self.x_data = x_data
        self.y_data = y_data
        self.idx_data = idx_data
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index], self.idx_data[index]
    def __len__(self): 
        return self.x_data.shape[0]


class TestDataset(Dataset):
    def __init__(self, x_data, idx_data):
        self.x_data = x_data
        self.idx_data = idx_data
    def __getitem__(self, index): 
        return self.x_data[index], self.idx_data[index]
    def __len__(self): 
        return self.x_data.shape[0]


dataDir = sys.argv[1] #"./dataset"
resDir = sys.argv[2] #"./results"
num_seq_length = int(sys.argv[3]) #3
pred_range_days = int(sys.argv[4]) #15
train_epochs = int(sys.argv[5]) #2000

os.makedirs(resDir, exist_ok = True)
train_rna_seq = pd.read_csv(os.path.join(dataDir, "train_gene_expression.csv"), index_col = 'timepoint')
train_cytokine = pd.read_csv(os.path.join(dataDir, "train_cytokine_expression.csv"), index_col = 'timepoint')

test_rna_seq = pd.read_csv(os.path.join(dataDir, "test_gene_expression.csv"), index_col = 'timepoint')

sample_id_col = 'sample_id'
test_cytokine_sample_id = test_rna_seq[sample_id_col].tolist()

del train_rna_seq[sample_id_col]
del train_cytokine[sample_id_col]
del test_rna_seq[sample_id_col]

n_feature_rna_seq = len(train_rna_seq.columns)
n_feature_cytokine = len(train_cytokine.columns)

train_size = int(len(train_rna_seq)/num_seq_length)


def build_train_time_series_dataset(raw_x, raw_y, seq_length = 3) :
    data_x = []
    data_y = []
    data_idx = []
    for i in range(0, len(raw_x), seq_length):
        _x = raw_x[i:i+seq_length]
        _y = raw_y[i:i+seq_length]
        timepoint_df = pd.DataFrame({'timepoint' : _x.index.tolist()})
        timepoint_df['timepoint'] = timepoint_df['timepoint'].apply(lambda x : datetime.strptime(x, "%Y-%m-%d"))
        _idx = [0]
        for j in range(1, len(timepoint_df)) :
            _idx.append((timepoint_df['timepoint'][j] - timepoint_df['timepoint'][0]).days)
        data_x.append(_x)
        data_y.append(_y)
        data_idx.append(_idx)
    return np.array(data_x), np.array(data_y), np.array(data_idx)


def build_test_time_series_dataset(raw_x, seq_length = 3) :
    data_x = []
    data_y = []
    data_idx = []
    for i in range(0, len(raw_x), seq_length):
        _x = raw_x[i:i+seq_length]
        timepoint_df = pd.DataFrame({'timepoint' : _x.index.tolist()})
        timepoint_df['timepoint'] = timepoint_df['timepoint'].apply(lambda x : datetime.strptime(x, "%Y-%m-%d"))
        _idx = [0]
        for j in range(1, len(timepoint_df)) :
            _idx.append((timepoint_df['timepoint'][j] - timepoint_df['timepoint'][0]).days)
        data_x.append(_x)
        data_idx.append(_idx)
    return np.array(data_x), np.array(data_idx)


train_x, train_y, train_idx = build_train_time_series_dataset(train_rna_seq, train_cytokine, num_seq_length)
test_x, test_idx = build_time_series_dataset(test_rna_seq, test_cytokine, num_seq_length)

train_x = torch.FloatTensor(train_x)
train_y = torch.FloatTensor(train_y)

test_x = torch.FloatTensor(test_x)

batch_size = 30
n_h1 = 1024
n_h2 = 512

train_dataset = TimeSeriesDataset(train_x, train_y, train_idx)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True) #drop_last=True

test_dataset = TestDataset(test_x, test_idx)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False) 


class TimeSeriesRegressNet(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.dense_layer = nn.Sequential(
            nn.Linear(n_feature_rna_seq, n_h1),
            nn.LeakyReLU(), #nn.LeakyReLU(),
            nn.Linear(n_h1, n_h2),
            nn.LeakyReLU(), #nn.LeakyReLU(),
            nn.Linear(n_h2, n_feature_cytokine),
            nn.LeakyReLU()
            )
    #
    def interpolateBlock(self, x) : #expr_ratio
        _x = torch.transpose(x, 1, 2)
        _x_res = F.interpolate(_x, size = pred_range_days, mode = "linear")
        _x_res = torch.transpose(_x_res, 1, 2)
        return _x_res
    #
    def forward(self, x) :
        _x = self.dense_layer(x)
        pred = self.interpolateBlock(_x)
        return pred


learning_rate = 1e-4
model = TimeSeriesRegressNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
mae_loss = nn.L1Loss()


def train_regressor(epoch, train_dataloader, model, mae_loss, optimizer) :
    model.train()
    total_loss = 0.0
    for batch, (x, y, idx_data) in enumerate(train_dataloader):
        x, y, idx_data = x.to(device), y.to(device), idx_data.to(device)
        pred = model(x)
        batch_mae_loss = 0.0
        for batch_idx in range(len(x)) :
            batch_mae_loss += mae_loss(pred[batch_idx, idx_data[batch_idx]], y[batch_idx])
        batch_mae_loss /= len(x)
        optimizer.zero_grad()
        batch_mae_loss.backward()
        optimizer.step()
        total_loss += batch_mae_loss
    if epoch % 10 == 0 :
        print(f"[Epoch] {epoch + 1}\tTraining loss : {total_loss:>5f}")


def test_regressor(model, test_dataloader) :
    model.eval()
    pred_list = []
    with torch.no_grad() :
        for batch, (x, idx_data) in enumerate(test_dataloader):
            x, idx_data = x.to(device), idx_data.to(device)
            pred = model(x)
            for batch_idx in range(len(x)) :
                tmp_pred = pred[batch_idx, idx_data[batch_idx]]
                pred_list.append(tmp_pred.detach().cpu().numpy())
    return pred_list


for t in range(train_epochs) :
    train_regressor(t, train_dataloader, model, mae_loss, optimizer)


pred_list = test_regressor(model, test_dataloader)
pred = np.array(pred_list)


pred = pred.reshape(-1, n_feature_cytokine)


pred_df = pd.DataFrame(pred)
pred_df.columns = train_cytokine.columns.tolist()
pred_df.index = test_cytokine_sample_id
pred_df['timepoint'] = test_rna_seq.index.tolist()

pred_df.to_csv(os.path.join(resDir, "pred_cytokine.csv"), mode = "w", index = True)








