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