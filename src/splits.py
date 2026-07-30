import numpy as np


def grouped_kfold(pairs_df, k=5, group_col="Name"):
    unique_groups = np.asarray(pairs_df[group_col].unique())

    rng = np.random.default_rng(seed=42)
    rng.shuffle(unique_groups)

    # assign each lifter to a fold 0..k-1, as evenly as possible
    group_to_fold = {name: i % k for i, name in enumerate(unique_groups)}

    # look up every row's fold via its lifter's assignment
    row_folds = pairs_df[group_col].map(group_to_fold).to_numpy()

    for fold in range(k):
        val_idx = np.where(row_folds == fold)[0]
        train_idx = np.where(row_folds != fold)[0]
        yield train_idx, val_idx
