import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.pairs import build_meet_pairs, train_test_split_by_lifter
from src.features import build_features
from src.models.neural_net import NeuralNetScratch

df = pd.read_csv('data/openpowerlifting_processed.csv', parse_dates=['Date'])
pairs = build_meet_pairs(df)
pairs = build_features(pairs)

FEATURE_COLS = [
    'TotalKg', 'BodyweightKg', 'Age', 'meet_number', 'days_since_last_meet',
    'prev_total_change', 'bodyweight_change', 'rolling_avg_change_last_3_meets',
    'linear_trend_slope_last_N_meets',
]

pairs = pairs.dropna(subset=FEATURE_COLS + ['next_TotalKg'])
train_df, test_df = train_test_split_by_lifter(pairs)

X_train = train_df[FEATURE_COLS].to_numpy()
y_train = train_df['next_TotalKg'].to_numpy()

# sigmoid saturates on raw feature scales (TotalKg in the hundreds) - standardize,
# same requirement as gradient descent back in Phase 2
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
X_train_scaled = (X_train - X_mean) / X_std

hidden_units = 8
learning_rate = 0.1
n_iterations = 1000

# --- our from-scratch model ---
model = NeuralNetScratch(hidden_units=hidden_units, learning_rate=learning_rate)
model.fit(X_train_scaled, y_train)

# --- equivalent tiny pytorch model, same architecture ---
torch.manual_seed(0)
n_features = X_train_scaled.shape[1]
torch_model = nn.Sequential(
    nn.Linear(n_features, hidden_units),
    nn.Sigmoid(),
    nn.Linear(hidden_units, 1),
)

X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

torch_loss_history = []
for _ in range(n_iterations):
    optimizer.zero_grad()
    predictions = torch_model(X_tensor)
    loss = loss_fn(predictions, y_tensor)
    loss.backward()
    optimizer.step()
    torch_loss_history.append(loss.item())

print("scratch final loss (MSE):", model.loss_history[-1])
print("pytorch final loss (MSE):", torch_loss_history[-1])

plt.plot(model.loss_history, label="scratch")
plt.plot(torch_loss_history, label="pytorch")
plt.xlabel("iteration")
plt.ylabel("loss (MSE)")
plt.title("NeuralNetScratch vs. PyTorch: training loss over iterations")
plt.legend()
plt.savefig("outputs/neural_net_loss_comparison.png")
