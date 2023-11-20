import torch
from torch import nn

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        super(LSTM_Model, self).__init__()
        '''
        input_dim = 輸入維度
        hidden_dim = 隱藏層維度
        num_layers = LSTM層數
        output_dim = 輸出維度
        '''
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim,output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        output, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        output = self.linear(output[:, -1, :])
        return output
    
class BiLSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,device):
        super(BiLSTM_Model, self).__init__()
        '''
        input_dim = 輸入維度
        hidden_dim = 隱藏層維度
        num_layers = LSTM層數
        output_dim = 輸出維度
        '''
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim, output_dim)
    def forward(self,x):
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        output, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        output = self.linear(output[:, -1, :])
        return output

class LSTM_MLP_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, mlp_hidden_dim, num_layers, output_dim, device, dropout=0):
        super(LSTM_MLP_Model, self).__init__()
        '''
        input_dim = 輸入維度
        lstm_hidden_dim = LSTM隱藏層維度
        mlp_hidden_dim = MLP隱藏層維度
        num_layers = LSTM層數
        output_dim = 輸出維度
        '''
        self.device = device
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim, mlp_hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim,output_dim)
        )
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_dim).to(self.device)
        output, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        output = self.mlp(output[:, -1, :])
        return output
    
class BiLSTM_MLP_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, mlp_hidden_dim, num_layers, output_dim, device, dropout=0,):
        super(BiLSTM_MLP_Model, self).__init__()
        '''
        input_dim = 輸入維度
        lstm_hidden_dim = 隱藏層維度
        mlp_hidden_dim = MLP隱藏層維度
        num_layers = LSTM層數
        output_dim = 輸出維度
        '''
        self.device = device
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(2*lstm_hidden_dim, mlp_hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim,output_dim)
        )
    def forward(self,x):
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.lstm_hidden_dim).to(self.device)
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.lstm_hidden_dim).to(self.device)
        output, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        output = self.mlp(output[:, -1, :])
        return output


class GRU_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,device):
        super(GRU_Model, self).__init__()
        '''
        input_dim = 輸入維度
        hidden_dim = 隱藏層維度
        num_layers = GRU層數
        output_dim = 輸出維度
        '''
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if len(x.shape) == 2:
            h0 = torch.zeros(self.num_layers, 1, self.hidden_dim).to(self.device)
            x = x.unsqueeze(0)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        
        out, hn = self.gru(x, h0.detach())
        out = self.linear(out[:, -1, :])
        return out
