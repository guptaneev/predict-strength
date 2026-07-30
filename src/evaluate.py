import numpy as np
import pandas as pd

from src.metrics import mae


def subgroup_mae(actual, predicted, group_labels) -> pd.DataFrame:
    df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted,
        'group': group_labels,
    })

    results = []
    for group_value, group_df in df.groupby('group'):
        results.append({
            'group': group_value,
            'mae': mae(group_df['actual'], group_df['predicted']),
            'count': len(group_df),
        })

    return pd.DataFrame(results)

def bootstrap_mae_ci(actual, predicted, n_boot=1000):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    n = len(actual)

    boot_maes = []
    for _ in range(n_boot):
        sample_idx = np.random.choice(n, size=n, replace=True)
        boot_maes.append(mae(actual[sample_idx], predicted[sample_idx]))

    low, high = np.percentile(boot_maes, [5, 95])
    return low, high