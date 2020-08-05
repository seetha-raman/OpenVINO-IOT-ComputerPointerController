import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from functools import reduce


def plot_metrics(metic):
    loadtime_dir = pathlib.Path(f'./results/{metic}').glob('*.csv')

    dfs = [pd.read_csv(path, index_col=0) for path in loadtime_dir]
    df = reduce(lambda df1, df2: df1.join(df2), dfs)
    print(df)

    plt.xticks(rotation=90)
    df.plot.bar(rot=0)

    # plt.show()
    plt.savefig(f'results/{metic}.png')


plot_metrics('loadtime')
plot_metrics('infertime')
