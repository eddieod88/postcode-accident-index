import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def true_vs_expected_histplot(y_true: pd.Series, y_pred: pd.Series, bins: int = 100):
    fig, ax = plt.subplots()
    sns.histplot(pd.concat([y_true, y_pred], axis=1), bins=bins, axes=ax)
    return fig
