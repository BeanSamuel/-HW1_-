import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn, optim
from lstm_model import LSTM_Model
from load_data import load_train_data

# Environment setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loading
input_date_data_size = 3
data_loader = load_train_data(folder_path='./Dataset/train', input_date_data_size=input_date_data_size, device=device)
train_x, train_y, valid_x, valid_y, feature_size = data_loader.train_x, data_loader.train_y, data_loader.valid_x, data_loader.valid_y, data_loader.feature_size

# Reshape data
train_x = train_x.reshape(-1, input_date_data_size, feature_size)
valid_x = valid_x.reshape(-1, input_date_data_size, feature_size)

# Model Initialization
model = LSTM_Model(input_dim=feature_size, hidden_dim=128, num_layers=1, output_dim=8, device=device).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_loss_history = []
val_loss_history = []

# Training
for epoch in tqdm(range(10000)):  # Using 10,000 epochs as an example
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

# Plotting only data from epoch 500 onwards
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history[500:], label='Training Loss', color='blue')
plt.plot(val_loss_history[500:], label='Validation Loss', color='red')
plt.title('Training vs Validation Loss from Epoch 500 onwards')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
