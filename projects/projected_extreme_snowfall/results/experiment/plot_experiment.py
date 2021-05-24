import os.path as op
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from projects.projected_extreme_snowfall.results.part_1.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.utils import load_combination_name_for_tuple
from root_utils import VERSION_TIME

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


def plot_utils(csv_filename, altitude):
    csv_filename += '.csv'
    csv_filepath = op.join(CSV_PATH, csv_filename)
    df = pd.read_csv(csv_filepath, index_col=0)
    # number of setting with missing combination
    ind_has_shape = df.index.str.contains('shape')
    df = df.loc[~ind_has_shape]
    print(len(df))

    ind = df.isnull().any(axis=0)
    df2 = df.loc[:, ind]
    print(df2.isnull().sum(axis=0))

    # Filter
    ind = df.isnull().any(axis=1)
    print('number of missing full combinations:', sum(ind))
    print('number of studied full combinations:', len(ind) - sum(ind))
    df = df.loc[~ind]

    column_name = 'sum' if altitude is None else 'sum_{}'.format(altitude)
    if altitude is not None:
        df = create_additional_sum_columns_for_elevation(df, altitude)

    ax = plt.gca()
    potential_indices = list(range(4))
    all_combinations = [potential_indices, potential_indices, [0]]
    combinations = list(product(*all_combinations))
    set_of_i_in_legend = set()

    no_effect_combination_value = None
    all_values_with_effect = []
    for combination in combinations:
        combination_name = load_combination_name_for_tuple(combination)
        try:
            value = -df.loc[combination_name, column_name]
            # Compute the abs
            nb_params = 0
            for i in combination:
                if i in [1, 3]:
                    nb_params += 6
                if i in [2, 3]:
                    nb_params += 11
            shift = 0.8
            for j, i in enumerate(combination):
                marker = 'o' if i == 0 else 'x'
                color = i_to_color[i]
                if i not in set_of_i_in_legend:
                    label = i_to_label[i]
                    set_of_i_in_legend.add(i)
                else:
                    label = None
                if combination == (0, 0, 0):
                    no_effect_combination_value = value
                    ax.hlines(value, linestyle='--', xmin=-5, xmax=35)
                else:
                    all_values_with_effect.append(value)
                ax.plot([nb_params + j * shift], [value], marker=marker, color=color,
                        linestyle='None', label=label)
        except KeyError:
            pass
    if no_effect_combination_value is not None:
        is_smaller_than_no_effect = [v < no_effect_combination_value for v in all_values_with_effect]
        percentage = np.round(100 * sum(is_smaller_than_no_effect) / len(is_smaller_than_no_effect))
        print('Percentages of effet combination lower than the "no effect combination" = {}\%'.format(percentage))
    ax.legend(loc='lower right')
    ax.set_xlabel('Number of effects')
    return ax


def plot_summary_calibration():
    year = 2015
    csv_filename = "fast_False_altitudes_1200_2100_3000_nb_of_models_27_nb_gcm_rcm_couples_20_splityear_{}".format(year)
    ax = plot_utils(csv_filename)
    start_year_end_test = csv_filename[-4:]
    ax.set_ylabel('Sum of nllh on the period {}-2019'.format(start_year_end_test))
    plt.show()


def plot_summary_calibration_with_model_as_truth():
    year = 2000
    csv_filename = "fast_False_altitudes_1200_2100_3000_nb_of_models_27_nb_gcm_rcm_couples_20_nb_samples_1_splityear_{}".format(
        year)
    ax = plot_utils(csv_filename)
    start_year_end_test = csv_filename[-4:]
    ax.set_ylabel('Sum of nllh on the period {}-2019'.format(start_year_end_test))
    plt.show()


def plot_summary_model_as_truth():
    csv_filename = "fast_False_altitudes_1200_2100_3000_nb_of_models_27_nb_gcm_rcm_couples_20_nb_samples_5_Vanoise"
    csv_filename = "fast_False_altitudes_1200_2100_3000_nb_of_models_27_nb_gcm_rcm_couples_20_nb_samples_5_Beaufortain"
    # csv_filename = "fast_False_altitudes_1200_2100_3000_nb_of_models_27_nb_gcm_rcm_couples_20_nb_samples_5_Haute-Tarentaise"
    # csv_filename = "fast_False_altitudes_1200_2100_3000_nb_of_models_27_nb_gcm_rcm_couples_20_nb_samples_5"
    for altitude in [None, 3000, 2100, 1200][:]:
        ax = plot_utils(csv_filename, altitude)
        end = 'the 3 altitudes' if altitude is None else 'the {} m'.format(altitude)
        title = 'model-as-truth experiment for the {}'.format(end)
        ax.set_ylabel('Log likelihood on the period 2020-2100\n'
                      'summed on the {}'.format(title))
        filename = "{}/{}".format(VERSION_TIME, title)
        StudyVisualizer.savefig_in_results(filename, transparent=False)
        plt.close()
        # plt.show()


def create_additional_sum_columns_for_elevation(df, altitude):
    altitude_str = str(altitude)
    columns_for_altitudes = [c for c in df.columns if altitude_str in c]
    new_column = 'sum_{}'.format(altitude_str)
    if new_column not in df:
        df_altitude = df.loc[:, columns_for_altitudes]
        ind = df_altitude.isnull().any(axis=1)
        df[new_column] = np.nan
        df[new_column] = df_altitude.sum(axis=1)
    return df


if __name__ == '__main__':
    # plot_summary_calibration()
    # plot_summary_calibration_with_model_as_truth()
    plot_summary_model_as_truth()
    # create_additional_sum_columns_for_elevation()
