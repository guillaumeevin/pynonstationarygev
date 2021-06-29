
import matplotlib.pyplot as plt
import os.path as op

import numpy as np
import pandas as pd

from extreme_trend.ensemble_simulation.visualizer_for_simulation_ensemble import SIMULATION_PATH
from projects.projected_extreme_snowfall.results.part_2.v2.utils import load_excel


def main_plot():
    excel_name = 'RMSE_ShiftExperiment_10_0'
    excel_filepath = op.join(SIMULATION_PATH, excel_name + '.xlsx')
    df = load_excel(excel_filepath, "Main")
    short_name_to_label_and_color = {
        "Together fit without": ("Together fit without", 'red'),
        "Together fit with": ("Together fit with", 'darkred'),
        "Separate fit with": ("Separate fit with", 'darkblue'),
        "Separate fit without": ("Separate fit without", 'blue')
    }
    ax = plt.gca()
    shift_list = [0, 10, 20]

    bincenters = np.array(list(range(len(df.columns)))) * 1.0
    df = df.iloc[[0, 2, 1, 3], :]
    for j, (i, row) in enumerate(df.iterrows()):
        label, color = short_name_to_label_and_color[i]
        label += " correction coefficients"
        y = []
        all_shift = []

        for c1 in shift_list:
            indices = ["({}, {})".format(c1, c2) for c2 in shift_list]
            values = [row.loc[i] for i in indices]
            y.extend(values)
            all_shift.extend(shift_list)
        ax.bar(bincenters + (j-1) * 0.1 - (0.1 / 2) , y, width=0.1, color=color, label=label)

        if j == 0:
            ax.set_xticklabels(all_shift)
            ax.set_xticks(bincenters)

    metric_name = ' '.join(excel_name.split('Shift')[0].split('_'))
    ylabel = metric_name + ' for the 100-year return level in 2100'
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Average bias in the Mean (%)")
    ax2 = ax.twiny()
    x_ticks = bincenters[1::len(shift_list)]
    print(x_ticks)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(shift_list)
    ax2.set_xlabel("Average bias in the Standard deviation (%)")
    ax.legend()
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim((ymin, ymax))
    ax2.set_xlim(ax.get_xlim())
    ax.vlines(np.mean(bincenters[2:4]), ymin, ymax)
    ax.vlines(np.mean(bincenters[5:7]), ymin, ymax)
    plt.show()


    print(df)

if __name__ == '__main__':
    main_plot()
