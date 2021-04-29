
import os.path as op
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from projects.projected_extreme_snowfall.results.part_1.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.utils import load_combination_name_for_tuple

i_to_color = {
    0: 'k',
    1: 'g',
    2: 'r',
    3: 'b'
}
i_to_label = {
    0: 'no effect',
    1: 'GCM effects',
    2: 'RCM effects',
    3: 'GCM and RCM effects'
}

def plot_summary():
    csv_filename = "fast_False_altitudes_1200_2100_3000_nb_of_models_27_nb_gcm_rcm_couples_20_nb_samples_1"
    csv_filename += '.csv'
    csv_filepath = op.join(CSV_PATH, csv_filename)
    df_csv = pd.read_csv(csv_filepath, index_col=0)

    ax = plt.gca()

    potential_indices = list(range(4))
    all_combinations = [potential_indices for _ in range(3)]
    combinations = list(product(*all_combinations))

    set_of_i_in_legend = set()
    for combination in combinations:
        combination_name = load_combination_name_for_tuple(combination)
        try:
            value = df_csv.loc[combination_name, 'sum']
            print('Found:', combination_name)
            # Compute the abs
            nb_params = 0
            for i in combination:
                if i in [1, 3]:
                    nb_params += 6
                if i in [2, 3]:
                    nb_params += 11
            shift = 1
            for j, i in enumerate(combination):
                marker = 'o' if i == 0 else 'x'
                color = i_to_color[i]
                if i not in set_of_i_in_legend:
                    label = i_to_label[i]
                    set_of_i_in_legend.add(i)
                else:
                    label = None
                ax.plot([nb_params + j * shift], [value], marker=marker, color=color,
                        linestyle='None', label=label)
        except KeyError:
            pass
    ax.set_xlabel('Number of parameters for the effects')
    ax.set_ylabel('Sum of nllh on the period 2020-2100')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    plot_summary()