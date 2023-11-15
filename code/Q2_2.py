import pickle
import matplotlib.pyplot as plt

with open('./output/loss_histories0.1.pkl', 'rb') as f:
    h01 = pickle.load(f)
with open('./output/loss_histories0.01.pkl', 'rb') as f:
    h001 = pickle.load(f)
with open('./output/loss_histories0.001.pkl', 'rb') as f:
    h0001 = pickle.load(f)
with open('./output/loss_histories0.0001.pkl', 'rb') as f:
    h00001 = pickle.load(f)
plt.plot(h01, label=f'LR={0.1}')
plt.plot(h001, label=f'LR={0.01}')
plt.plot(h0001, label=f'LR={0.001}')
plt.plot(h00001, label=f'LR={0.0001}')


plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Convergence of Different Learning Rates')
plt.show()
