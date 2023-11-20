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
import pickle
#Env
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATA
input_date_data_size = 3
data_loader = load_train_data(folder_path='./Dataset/train', input_date_data_size=input_date_data_size, device=device,do_mean=False)
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
model = model_choices[model_name].to(device)
if 'LSTM' in model_name or model_name == 'attention' or model_name == 'GRU':
    train_x = train_x.reshape(-1, input_date_data_size, feature_size)
    valid_x = valid_x.reshape(-1, input_date_data_size, feature_size)
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=100,verbose=True,enabled=False)

#TRAIN
warmup_epochs = 3000

train_loss_history = []
val_loss_history = []

print("Start Training -- Warming Up")
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in tqdm(range(warmup_epochs)):
    model.train()
    train_y_predicted = model(train_x)
    loss = criterion(train_y_predicted, train_y)
    train_loss_history.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    model.eval()
    val_y_predicted = model(valid_x)
    val_loss = criterion(val_y_predicted, valid_y)
    val_loss_history.append(val_loss.item())
    if (epoch + 1) % 1000 == 0:
        print(f'epoch: {epoch+1}, train_loss = {loss.item(): .6f}, val_loss = {val_loss.item(): .6f}')
    if early_stopping.early_stop:
        epochs = epoch
        print("Early stopping")
        print(f'epoch: {epoch+1}, train_loss = {loss.item(): .6f}, val_loss = {val_loss.item(): .6f}')
        break

with open('val_predictions_non_normalized.pkl', 'wb') as f:
    pickle.dump(val_loss_history, f)