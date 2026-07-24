from src.data import load_clean_data
from src.pairs import build_meet_pairs, train_test_split_by_lifter
from src.models.baseline import BaselineModel
from src.metrics import mae

df = load_clean_data()
pairs = build_meet_pairs(df)

train_df, test_df = train_test_split_by_lifter(pairs)

model = BaselineModel()
model.fit(train_df)
predictions = model.predict(test_df)

print(mae(test_df['next_TotalKg'], predictions))

"""
so we're basically just taking every single meet
and then shifting it down one to get this "next_totalkg"
which is the next total (aka the actual number)

our predictions are made by  going through each delta
between meets and averaging them all as a running
"avg_past_delta" and then using that to predict our lifters
next meet

the very first "avg_past_delta" will always be NaN because there is no
delta for the very first meet (i.e. meets 0->1)

our fitting is a mock model training by using the entire training
dataset's "delta" and averaging those to get a 
"population_avg_delta" which is used when we don't have a lifter's
"avg_past_delta"

prediction is made by adding "avg_past_delta" to the current "totalkg"

MAE is better because we don't want large changes / outliers
to be magnified even more than they already are. a 50kg jump
would have an impact of 2500kg (!!) with an MSE

predict zero change baseline would essentially just have the average 
absolute value of delta between meets as the MAE. this is not good to
compare against because it has no real reasoning, so even the worst
models can perform better than it. while this model is still simple,
it's data backed and has real reasoning behind its predictions.

"""

