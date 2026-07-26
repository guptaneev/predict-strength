import matplotlib.pyplot as plt

from src.data import load_clean_data
from src.metrics import mae
from src.pairs import build_meet_pairs, train_test_split_by_lifter
from src.models.linear_regression import LinearRegressionScratch
from sklearn.linear_model import LinearRegression

df = load_clean_data()
pairs = build_meet_pairs(df)

train_df, test_df = train_test_split_by_lifter(pairs)

# ŷ = w1·TotalKg + w2·avg_past_delta + b

# fill missing avg_past_delta (first transition per lifter) using train-only stats
population_avg_delta = train_df['delta'].mean()
train_df = train_df.copy()
test_df = test_df.copy()
train_df['avg_past_delta'] = train_df['avg_past_delta'].fillna(population_avg_delta)
test_df['avg_past_delta'] = test_df['avg_past_delta'].fillna(population_avg_delta)

X_train = train_df[['TotalKg', 'avg_past_delta']].to_numpy()
y_train = train_df['next_TotalKg'].to_numpy()

X_test = test_df[['TotalKg', 'avg_past_delta']].to_numpy()
y_test = test_df['next_TotalKg'].to_numpy()

# standardize features (train-only mean/std, reused for test - same leakage rule as everywhere else)
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train_scaled = (X_train - X_train_mean) / X_train_std
X_test_scaled = (X_test - X_train_mean) / X_train_std

'''
model = LinearRegressionScratch(method="normal_equation")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

sklearnModel = LinearRegression()
sklearnModel.fit(X_train, y_train)
sklearn_predictions = sklearnModel.predict(X_test)

print("scratch weights (w1, w2, b):", model.weights)
print("sklearn weights (w1, w2):", sklearnModel.coef_, "bias:", sklearnModel.intercept_)
print("scratch MAE:", mae(y_test, predictions))
print("sklearn MAE:", mae(y_test, sklearn_predictions))

'''



model = LinearRegressionScratch(method="gradient_descent", num_iterations=2000, learning_rate=0.1)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
print("gradient descent weights (scaled-feature space):", model.weights)
print("gradient descent MAE:", mae(y_test, predictions))

plt.plot(model.loss_history)
plt.xlabel("iteration")
plt.ylabel("loss (MSE)")
plt.title("Gradient descent loss over iterations")
plt.savefig("outputs/gradient_descent_loss.png")

plt.figure()
zoom_n = 50
plt.plot(model.loss_history[:zoom_n])
plt.xlabel("iteration")
plt.ylabel("loss (MSE)")
plt.title(f"Gradient descent loss - first {zoom_n} iterations")
plt.savefig("outputs/gradient_descent_loss_zoomed.png")