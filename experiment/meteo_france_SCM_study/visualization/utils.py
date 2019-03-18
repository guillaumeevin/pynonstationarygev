import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as tkr  # has classes for tick-locating and -formatting


def plot_df(df: pd.DataFrame, ax=None, ylabel='Example plot', show=True, xlabel='Altitude (m)'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for _, row in df.iterrows():
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(np.array(row.index), row.values, '-+', label=row.name)
        ax.set_title("")
        ax.legend()

    if show:
        plt.show()


def example_plot_df():
    df = pd.DataFrame.from_dict({
        '1800': [0, 2],
        '2400': [1, 3],
    }, columns=['North', 'South'], orient='index').transpose()
    plot_df(df)


def get_km_formatter():
    # From: https://stackoverflow.com/questions/27575257/how-to-divide-ytics-to-a-certain-number-in-matplotlib
    def numfmt(x, pos):  # your custom formatter function: divide by 1000.0
        s = '{}'.format(int(x / 1000.0))
        return s

    return tkr.FuncFormatter(numfmt)  # create your custom formatter function


if __name__ == '__main__':
    example_plot_df()
