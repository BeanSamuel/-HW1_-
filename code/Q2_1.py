from torch import (
    nn,
    optim,
)

from early_stopping import (
    EarlyStopping,
)

from lstm_model import (
    LSTM_Model,
    BiLSTM_Model,
    LSTM_MLP_Model,
    BiLSTM_MLP_Model,
    GRU_Model,
)

from attention_model import(
    TimeSeriesAttentionModel,
)

from load_data import (
    load_train_data,
    load_test_data,
)

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

#Env
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATA
input_date_data_size = 3
data_loader = load_train_data(folder_path='./Dataset/train', input_date_data_size=input_date_data_size, device=device)
train_x, train_y, valid_x, valid_y, feature_size = data_loader.train_x, data_loader.train_y, data_loader.valid_x, data_loader.valid_y, data_loader.feature_size

#MODEL_CHOICE
model_choices = {
    'LSTM': LSTM_Model(input_dim=feature_size, hidden_dim=128, num_layers=1, output_dim=8,device=device),
    'BILSTM': BiLSTM_Model(input_dim=feature_size, hidden_dim=128, num_layers=1, output_dim=8,device=device),
    'LSTM_MLP': LSTM_MLP_Model(input_dim=feature_size, lstm_hidden_dim=128, mlp_hidden_dim=64, num_layers=1, output_dim=8, dropout=0.1,device=device),
    'BILSTM_MLP': BiLSTM_MLP_Model(input_dim=feature_size, lstm_hidden_dim=128, mlp_hidden_dim=64, num_layers=1, output_dim=8, dropout=0.1,device=device),
    'attention': TimeSeriesAttentionModel(input_dim=feature_size,embed_size=16,heads=4,num_layers=4,output_dim=8,dropout=0.1),
    'GRU': GRU_Model(input_dim=feature_size, hidden_dim=128, num_layers=2, output_dim=8,device=device)
}

#PARAMETER 
model_name = 'LSTM'
if 'LSTM' in model_name or model_name == 'attention' or model_name == 'GRU':
    train_x = train_x.reshape(-1, input_date_data_size, feature_size)
    valid_x = valid_x.reshape(-1, input_date_data_size, feature_size)
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=100,verbose=True,enabled=False)

#TRAIN

lr = 0.0001
learning_rates = [lr]
loss_histories = {}

model = model_choices[model_name].to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
optimizer.zero_grad()
loss_histories[lr] = []

for epoch in tqdm(range(1000)):
    model.train()
    train_y_predicted = model(train_x)
    loss = criterion(train_y_predicted, train_y)
    loss_histories[lr].append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

import pickle

with open(f'./output/loss_histories{lr}.pkl', 'wb') as f:
    pickle.dump(loss_histories[lr], f)