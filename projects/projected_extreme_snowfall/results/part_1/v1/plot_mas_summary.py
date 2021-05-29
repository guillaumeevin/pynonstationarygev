import os.path as op
from itertools import product

import matplotlib.pyplot as plt

import pandas as pd

from projects.projected_extreme_snowfall.results.combination_utils import climate_coordinates_with_effects_list, \
    load_combination_name_for_tuple
from projects.projected_extreme_snowfall.results.part_1.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.part_1.v1.utils_v1 import load_combination_name_to_dict_v2

def plot_summary_graph_for_w():
    # csv_filename = "nbloop{}_fast_None_altitudes_600_2100_3600_nb_of_models_27_nb_gcm_rcm_couples_20_nb_samples_6.csv"
    csv_filename = "nbloop{}_fast_None_altitudes_2100_3600_nb_of_models_27_nb_gcm_rcm_couples_20_nb_samples_3.csv"
    combinations = [(i, i, i) for i in range(4)][:]
    combinations_names = list(load_combination_name_to_dict_v2(climate_coordinates_with_effects_list,
                                                                combinations).keys())
    combination_name_to_res = {c: [] for c in combinations_names}
    w_list = []
    for w in range(1, 40):
        csv_filepath = op.join(CSV_PATH, csv_filename.format(w))
        if op.exists(csv_filepath):
            print('w=', w)
            w_list.append(w)
            df_csv = pd.read_csv(csv_filepath, index_col=0)
            for combination_name in combinations_names:
                try:
                    value = df_csv.loc[combination_name, 'min']
                except KeyError:
                    value = None
                print(combination_name, value)
                combination_name_to_res[combination_name].append(value)
    # Plot
    ax = plt.gca()
    for combination_name, res in combination_name_to_res.items():
        w_list_for_plot = [w for i, w in enumerate(w_list) if res[i] is not None]
        res_for_plot = [r for r in res if r is not None]
        ax.plot(w_list_for_plot, res_for_plot, label=combination_name.replace('_', ' '),
                marker='x')
    ax.set_xlabel('w')
    ax.set_ylabel('Minimum averaged nllh on the period 2020-2100')
    ax.legend()
    plt.show()

def plot_summary_graph_for_fixed_w(w=1):
    csv_filename = "nbloop{}_fast_None_altitudes_2100_3600_nb_of_models_27_nb_gcm_rcm_couples_20_nb_samples_3"
    # csv_filename = "nbloop{}_fast_None_altitudes_600_2100_3600_nb_of_models_27_nb_gcm_rcm_couples_20_nb_samples_6"
    csv_filename = csv_filename.format(w) + '.csv'
    csv_filepath = op.join(CSV_PATH, csv_filename.format(w))
    df_csv = pd.read_csv(csv_filepath, index_col=0)

    combinations = [(0,0,0), (3,3,3)]

    potential_indices = list(range(4))
    all_combinations = [potential_indices for _ in range(3)]
    combinations = list(product(*all_combinations))
    ax = plt.gca()
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
    set_of_i_in_legend = set()
    for combination in combinations:
        combination_name = load_combination_name_for_tuple(combination)
        try:
            value = df_csv.loc[combination_name, 'min']
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
    ax.set_ylabel('Minimum averaged nllh on the period 2020-2100')
    ax.legend()
    plt.show()



if __name__ == '__main__':
    plot_summary_graph_for_w()
    # plot_summary_graph_for_fixed_w(w=1)