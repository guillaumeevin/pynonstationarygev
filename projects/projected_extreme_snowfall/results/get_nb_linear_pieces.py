import os
from collections import Counter

import matplotlib.pyplot as plt
import os.path as op

import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from projects.projected_extreme_snowfall.results.seleciton_utils import get_short_name, short_name_to_label, \
    short_name_to_color, number_to_model_name, number_to_model_class, short_name_to_parametrization_number
from projects.projected_extreme_snowfall.results.setting_utils import get_last_year_for_the_train_set

numbers_of_pieces = list(range(1, 5))

model_as_truth_excel_folder = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/ModelAsTruthExperiment/2100 snowfall"
calibration_aic_excel_folder = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/CalibrationAicExperiment/2100 snowfall"
calibration_excel_folder = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/CalibrationValidationExperiment/snowfall_2100_{}"


def run_selection(massif_names, min_mode=False, show=False):
    d = massif_name_to_nb_linear_pieces(massif_names, min_mode)

    massif_name_to_short_name = {}
    for massif_name, number in d.items():
        percentages = [0.6, 0.7, 0.8][:]
        df = pd.DataFrame()
        for p in percentages:
            year = get_last_year_for_the_train_set(p)
            excel_folder = calibration_excel_folder.format(year)
            df2 = load_df_complete(massif_name, [number], excel_folder)
            df = pd.concat([df, df2], axis=1)
        # Compute the mean
        s = df.mean(axis=1)
        short_name = get_short_name(s.idxmin())
        massif_name_to_short_name[massif_name] = short_name

    # Plot the map
    # ax = plt.gca()
    # AbstractStudy.visualize_study(ax, massif_name_to_color={m: short_name_to_color[s] for m, s in
    #                                                         massif_name_to_short_name},
    #                               massif_name_to_text={m: str(v) for m, v in d.items()},
    #                               add_text=True,
    #                               axis_off=True)
    # plt.show()
    # plt.close()

    if show:
        plots(massif_name_to_short_name)

    massif_name_to_model_class = {m: number_to_model_class[n] for m, n in d.items()}
    massif_name_to_parametrization_number = {m: short_name_to_parametrization_number[s] for m, s in massif_name_to_short_name.items()}
    return massif_name_to_model_class, massif_name_to_parametrization_number


def plots(massif_name_to_short_name):
    # plot the histogram for the number of pieces
    # c = Counter(d.values())
    # number_of_occurences = [c[n] if n in c else 0 for n in numbers_of_pieces]
    # percentage = [100 * n / sum(number_of_occurences) for n in number_of_occurences]
    # ax = plt.gca()
    # ax.bar(numbers_of_pieces, percentage)
    # ax.set_xticks(numbers_of_pieces)
    # ax.set_ylabel('Percentage of massifs (%)')
    # ax.set_xlabel('Number of linear pieces that minimizes the mean log score')
    # plt.show()
    # plot the histogram for the parametrization
    labels = [short_name_to_label[s] for s in massif_name_to_short_name.values()]
    c = Counter(labels)
    ordered_short_name = [
        "no effect",
        "is_ensemble_member",
        "gcm",
        "rcm",
        "gcm_and_rcm",

    ]
    ordered_labels = [short_name_to_label[s] for s in ordered_short_name]
    ordered_colors = [short_name_to_color[s] for s in ordered_short_name]
    number_of_occurences = [c[n] if n in c else 0 for n in ordered_labels]
    percentage = [100 * n / sum(number_of_occurences) for n in number_of_occurences]
    ax = plt.gca()
    ordered_short_labels = [e.replace(' adjustment coefficient', '') for e in ordered_labels]
    ax.bar(ordered_short_labels, percentage, color=tuple(ordered_colors))
    ax.set_xticks(ordered_short_labels)
    ax.set_ylabel('Percentage of massifs (%)')
    ax.set_xlabel('Parameterization that minimizes the mean log score')
    plt.show()


def massif_name_to_nb_linear_pieces(massif_names, min_mode=True):
    massif_name_to_nb_linear_pieces = {}
    for massif_name in massif_names:
        nb1 = _get_nb_linear_pieces(massif_name, model_as_truth_excel_folder)
        if min_mode:
            nb2 = _get_nb_linear_pieces(massif_name, calibration_aic_excel_folder)
            updated_nb = min(nb1, nb2)
        else:
            updated_nb = nb1
        massif_name_to_nb_linear_pieces[massif_name] = updated_nb
    return massif_name_to_nb_linear_pieces


def _get_nb_linear_pieces(massif_name, excel_folder, ):
    s = load_serie_parametrization_without_adjustement(massif_name, numbers_of_pieces, excel_folder)
    return s.idxmin()


def load_serie_parametrization_without_adjustement(massif_name, numbers_of_pieces, excel_folder):
    df = load_df_complete(massif_name, numbers_of_pieces, excel_folder)
    return df.iloc[0]


def load_df_complete(massif_name, numbers_of_pieces, excel_folder):
    df = pd.DataFrame()
    for number_of_pieces in numbers_of_pieces:
        s = _load_dataframe(massif_name, number_of_pieces, excel_folder)
        df = pd.concat([df, s], axis=1)
    df = df.iloc[[2 * i + 1 for i in range(5)]]
    df.columns = numbers_of_pieces
    return df


def _load_dataframe(massif_name, number_of_pieces, excel_folder):
    model_name = number_to_model_name[number_of_pieces]
    assert op.exists(excel_folder)
    files = [f for f in os.listdir(excel_folder) if model_name in f]
    assert len(files) == 1
    filepath = op.join(excel_folder, files[0])
    df = pd.read_excel(filepath)
    columns = [c for c in df.columns if massif_name in c]
    s = df.loc[:, columns].mean(axis=1).iloc[2:]
    df = pd.DataFrame.from_dict({number_of_pieces: s})
    return df



if __name__ == '__main__':
    # massif_name_to_nb_linear_pieces(AbstractStudy.all_massif_names())
    massif_name_to_nb_linear_pieces_and_parametrization(AbstractStudy.all_massif_names(),
                                                        min_mode=False)
    # massif_name_to_nb_linear_pieces_and_parametrization(['Vanoise'])
