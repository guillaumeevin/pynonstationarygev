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


def create_adjusted_axes(nb_rows, nb_columns, figsize=(16, 10), subplot_space=0.5):
    fig, axes = plt.subplots(nb_rows, nb_columns, figsize=figsize)
    fig.subplots_adjust(hspace=subplot_space, wspace=subplot_space)
    return axes


def align_yaxis(ax1, v1, ax2, v2):
    # From https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)


def align_yaxis_on_zero(ax1, ax2):
    align_yaxis(ax1, 0, ax2, 0)


if __name__ == '__main__':
    example_plot_df()
