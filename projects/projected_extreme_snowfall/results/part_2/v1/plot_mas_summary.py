import os.path as op
from itertools import product

import matplotlib.pyplot as plt

import pandas as pd

from projects.projected_extreme_snowfall.results.combination_utils import climate_coordinates_with_effects_list, \
    load_combination_name_for_tuple
from projects.projected_extreme_snowfall.results.part_2.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.part_2.v1.utils_v1 import load_combination_name_to_dict_v2
from projects.projected_extreme_snowfall.results.part_2.v2.utils import main_sheet_name

def plot_summary_graph_for_fixed_w(w=1):
    csv_filename = "last_snow_load_fast_False_altitudes_1500_nb_of_models_1_nb_gcm_rcm_couples_20_alpha_3"
    csv_filename = "last_snow_load_fast_False_altitudes_1500_nb_of_models_1_nb_gcm_rcm_couples_20_alpha_3_selection_split_sample"
    # csv_filename = "last_snow_load_fast_None_altitudes_3000_nb_of_models_27_nb_gcm_rcm_couples_20_alpha_"
    csv_filename = csv_filename.format(w) + '.xlsx'
    csv_filepath = op.join(CSV_PATH, csv_filename.format(w))
    sheet_names = [main_sheet_name, main_sheet_name + " early", main_sheet_name + " later"]
    for sheet_name in sheet_names[:1]:
        df_csv = pd.read_excel(csv_filepath, index_col=0, sheet_name=sheet_name)
        name_columns = df_csv.columns
        name_columns = ['sum']
        for name_column in name_columns:
            print(name_column)

            potential_indices = [0, 1, 2, 3, 4]
            all_combinations = [potential_indices for _ in range(2)] + [[0]]
            combinations = list(product(*all_combinations))
            ax = plt.gca()
            i_to_color = {
                0: 'k',
                1: 'g',
                2: 'r',
                3: 'b',
                4: 'y'
            }
            i_to_label = {
                0: 'without correction coefficients',
                1: 'correction coefficients shared across GCMs',
                2: 'correction coefficients shared across RCMs',
                3: 'with correction coefficients shared across GCMs and RCMs',
                4: 'with one correction coefficient for each GCM-RCM pair'
            }
            set_of_i_in_legend = set()
            for combination in combinations:
                combination_name = load_combination_name_for_tuple(combination)
                try:
                    value = df_csv.loc[combination_name, name_column]
                    # Compute the abs
                    nb_params = 0
                    for i in combination:
                        if i in [1, 3]:
                            nb_params += 6
                        if i in [2, 3]:
                            nb_params += 11
                        if i in [4]:
                            nb_params += 19
                    shift = 0.8
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
            ax.set_xlabel('Number of correction coefficients')
            ax.set_ylabel('Sum of log likelihood on the period 2020-2100')
            a, b = ax.get_ylim()
            ax.set_ylim(1.02 * a, b)
            a, b = ax.get_xlim()
            value = df_csv.loc[load_combination_name_for_tuple((0, 0, 0)), name_column]
            ax.hlines(value, xmin=a, xmax=b, linestyle='--')
            ax.legend(prop={'size': 10})
            plt.show()
            plt.close()


# def plot_summary_graph_for_w():
#     # csv_filename = "nbloop{}_fast_None_altitudes_600_2100_3600_nb_of_models_27_nb_gcm_rcm_couples_20_nb_samples_6.csv"
#     csv_filename = "last_snow_load_fast_False_altitudes_1500_nb_of_models_8_nb_gcm_rcm_couples_20_alpha_40"
#     combinations = [(i, i, i) for i in range(4)][:]
#     combinations_names = list(load_combination_name_to_dict_v2(climate_coordinates_with_effects_list,
#                                                                 combinations).keys())
#     combination_name_to_res = {c: [] for c in combinations_names}
#     w_list = []
#     for w in range(1, 40):
#         csv_filepath = op.join(CSV_PATH, csv_filename.format(w))
#         if op.exists(csv_filepath):
#             print('w=', w)
#             w_list.append(w)
#             df_csv = pd.read_csv(csv_filepath, index_col=0)
#             for combination_name in combinations_names:
#                 try:
#                     value = df_csv.loc[combination_name, 'min']
#                 except KeyError:
#                     value = None
#                 print(combination_name, value)
#                 combination_name_to_res[combination_name].append(value)
#     # Plot
#     ax = plt.gca()
#     for combination_name, res in combination_name_to_res.items():
#         w_list_for_plot = [w for i, w in enumerate(w_list) if res[i] is not None]
#         res_for_plot = [r for r in res if r is not None]
#         ax.plot(w_list_for_plot, res_for_plot, label=combination_name.replace('_', ' '),
#                 marker='x')
#     ax.set_xlabel('w')
#     ax.set_ylabel('Minimum averaged nllh on the period 2020-2100')
#     ax.legend()
#     plt.show()




if __name__ == '__main__':
    plot_summary_graph_for_fixed_w()