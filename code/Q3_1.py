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

#MODEL_CHOICE
model_choices = {
    'LSTM': LSTM_Model,
    'BILSTM': BiLSTM_Model,
    'LSTM_MLP': LSTM_MLP_Model,
    'BILSTM_MLP': BiLSTM_MLP_Model,
    'attention': TimeSeriesAttentionModel,
    'GRU': GRU_Model
}

model_name = 'LSTM'
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=100, verbose=True, enabled=True)
optimizer_choice = optim.Adam

data_sizes = [2]  # You can adjust this as per your requirement

for input_date_data_size in data_sizes:
    # LOAD DATA
    data_loader = load_train_data(folder_path='./Dataset/train', input_date_data_size=input_date_data_size, device=device)
    train_x, train_y, valid_x, valid_y, feature_size = data_loader.train_x, data_loader.train_y, data_loader.valid_x, data_loader.valid_y, data_loader.feature_size

    # Reshape data if necessary
    if 'LSTM' in model_name or model_name == 'attention' or model_name == 'GRU':
        train_x = train_x.reshape(-1, input_date_data_size, feature_size)
        valid_x = valid_x.reshape(-1, input_date_data_size, feature_size)

    # Model Initialization
    model = model_choices[model_name](input_dim=feature_size, hidden_dim=128, num_layers=1, output_dim=8, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #TRAIN
    train_loss_history = []
    val_loss_history = []

    for epoch in tqdm(range(1000)):
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

        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Predict on validation data after training
    model.eval()
    val_predictions = model(valid_x)

    # Save the results using pickle
    results = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "val_predictions": val_predictions.detach().cpu().numpy()
    }
    with open(f"./output/results_data_size_{input_date_data_size}.pkl", "wb") as file:
        pickle.dump(results, file)
