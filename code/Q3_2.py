import pickle
import matplotlib.pyplot as plt
import pandas as pd

data_sizes = [2, 4]
loss_histories = {}
val_predictions = {}

# Load results
for data_size in data_sizes:
    with open(f"./output/results_data_size_{data_size}.pkl", "rb") as file:
        results = pickle.load(file)
        loss_histories[data_size] = results["val_loss"]
        val_predictions[data_size] = results["val_predictions"]

# If you want to see the predictions in a tabular form
dfs = []
for key, val in val_predictions.items():
    temp_df = pd.DataFrame(val)
    temp_df['Data_Size'] = key
    dfs.append(temp_df)

df = pd.concat(dfs, axis=0).reset_index(drop=True)
print(df)


# Plot validation loss
plt.figure(figsize=(10,6))
for data_size, history in loss_histories.items():
    plt.plot(history, label=f"Data size: {data_size}")
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('Validation Loss for Different Data Sizes')
plt.show()


