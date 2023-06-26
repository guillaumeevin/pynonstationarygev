
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from projected_extremes.reviewing.reviewing_utils import load_csv_filepath_gof, mode_to_name


def plot_pvalue_test():
    ax = plt.gca()
    all_massif = True
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
    # for mode in range(14):
    for mode in [9, 10, 11, 12, 13, 14, 15, 16, 17]:
        percentages = []
        for altitude in altitudes:
            csv_filepath = load_csv_filepath_gof(mode, altitude, all_massif)
            s = pd.read_csv(csv_filepath, index_col=0)
            pvalues = s.iloc[:, 0].values
            count_above_5_percent = [int(m >= 0.05) for m in pvalues]
            percentage_above_5_percent = 100 * sum(count_above_5_percent) / len(count_above_5_percent)
            print(percentage_above_5_percent)
            percentages.append(100 - percentage_above_5_percent)
        ax.plot(percentages, altitudes, label=mode_to_name[mode].replace('fulleffect', 'oneeffect'))
    # ax.set_xlim((0, 100))
    # ylim = ax.get_ylim()
    # ax.vlines(5, ymin=ylim[0], ymax=ylim[1], color='k', linestyles='dashed', label='5\% significance level')
    ax.set_xlabel('% of failed Anderson-Darling test for a 5% significance level')
    ax.set_ylabel('Elevation (m)')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    plot_pvalue_test()