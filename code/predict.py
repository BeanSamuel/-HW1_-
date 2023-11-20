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

import torch
import pandas as pd

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
model = model_choices[model_name].to(device)
model.load_state_dict(torch.load('./model/model1.pth'))
model.eval()

#PREDICT
data_loader = load_test_data(folder_path='./Dataset/test', input_date_data_size=input_date_data_size,mean_x=data_loader.mean_x,std_x=data_loader.std_x,device=device)
test_x, feature_size = data_loader.test_x, data_loader.feature_size
if 'LSTM' in model_name or model_name == 'attention' or model_name=='GRU':
    test_x = test_x.reshape(-1, input_date_data_size, feature_size)

with torch.no_grad():
    predicted = model(test_x)
ids = [x for x in range(len(predicted))]
output_df = pd.DataFrame({'id': ids})
currency_columns = ["AUD", "CAD", "EUR", "GBP", "HKD", "JPY", "KRW", "USD"]

for i, column_name in enumerate(currency_columns):
    output_df[column_name] = [x[i] for x in predicted.tolist()]

output_df = output_df.clip(lower=0)
save_path = './output/test.csv'
output_df.to_csv(save_path, index=False)
print(output_df.head(5))