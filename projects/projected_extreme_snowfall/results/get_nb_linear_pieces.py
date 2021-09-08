import os
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt
import os.path as op

import pandas as pd
from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from projects.projected_extreme_snowfall.results.seleciton_utils import get_short_name, short_name_to_label, \
    short_name_to_color, number_to_model_name, number_to_model_class, short_name_to_parametrization_number
from projects.projected_extreme_snowfall.results.setting_utils import get_last_year_for_the_train_set
from root_utils import VERSION_TIME

numbers_of_pieces = list(range(0, 5))

model_as_truth_excel_folder = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/ModelAsTruthExperiment/{} {}"
calibration_aic_excel_folder = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/CalibrationAicExperiment/{} {}"
calibration_excel_folder = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/CalibrationValidationExperiment/{}_{}_{}"


def run_selection(massif_names, altitude, min_mode=False, show=False,
                  snowfall=False):
    snowfall_str = 'snowfall' if snowfall else "snow load"
    d = massif_name_to_nb_linear_pieces(massif_names, min_mode,
                                        altitude,
                                        snowfall_str)

    massif_name_to_short_name = {}
    print('\n\nstart table with split-sample:')
    for massif_name, number in d.items():
        print(massif_name.replace('_', ' '), end=' & ')
        percentages = [0.6, 0.7, 0.8][:]
        df = pd.DataFrame()
        for p in percentages:
            year = get_last_year_for_the_train_set(p)
            excel_folder = calibration_excel_folder.format(snowfall_str, altitude, year)
            df2 = load_df_complete(massif_name, [number], excel_folder)
            df = pd.concat([df, df2], axis=1)
        # Compute the mean
        s = df.mean(axis=1)
        best_idx = s.idxmin()

        # display for some table
        s2_index = list(s.index)
        s2_index[1:1] = s2_index[-1:]
        s2_index = s2_index[:-1]
        s2_values = list(s.values)
        s2_values[1:1] = s2_values[-1:]
        s2_values = s2_values[:-1]
        s2 = pd.Series(index=s2_index, data=s2_values)
        l = [str(round(v, 2)) for v in s2.values]
        print(
            " & ".join(["\\textbf{" + number + '}' if idx == best_idx else number for idx, number in zip(s2.index, l)]),
            '\\\\')
        short_name = get_short_name(best_idx)
        massif_name_to_short_name[massif_name] = short_name
    print('\n\nend with split-sample:')

    plots(massif_name_to_short_name, d, show)

    massif_name_to_model_class = {m: number_to_model_class[n] for m, n in d.items()}
    massif_name_to_parametrization_number = {m: short_name_to_parametrization_number[s] for m, s in
                                             massif_name_to_short_name.items()}
    return massif_name_to_model_class, massif_name_to_parametrization_number


def plots(massif_name_to_short_name, d, show):
    folder = op.join(VERSION_TIME, "selection method projections")

    # plot legend
    ax = plt.gca()

    short_names = ["no effect", "is_ensemble_member", "gcm", "rcm", "gcm_and_rcm"]
    legend_elements = []
    for short_name in short_names:
        color = short_name_to_color[short_name]
        label = short_name_to_label[short_name]
        line2d = Line2D([0], [0], color=color, lw=5, label=label)
        legend_elements.append(line2d)

    ax.legend(handles=legend_elements, prop={'size': 9})
    plot("legend", folder, show)

    # Plot the map
    ax = plt.gca()
    massif_name_to_color = {m: short_name_to_color[s] for m, s in massif_name_to_short_name.items()}
    AbstractStudy.visualize_study(ax, massif_name_to_color=massif_name_to_color,
                                  massif_name_to_text={m: str(v) for m, v in d.items()},
                                  add_text=True,
                                  axis_off=True, show=False)

    plot("map", folder, show)

    # plot the histogram for the number of pieces
    c = Counter(d.values())
    number_of_occurences = [c[n] if n in c else 0 for n in numbers_of_pieces]
    percentage = [100 * n / sum(number_of_occurences) for n in number_of_occurences]
    ax = plt.gca()
    ax.bar(numbers_of_pieces, percentage)
    ax.set_xticks(numbers_of_pieces)
    ax.set_ylabel('Percentage of massifs (%)')
    ax.set_xlabel('Number of linear pieces that minimizes the mean log score')

    plot("histo number pieces", folder, show)

    # plot the histogram for the parametrization
    ax = plt.gca()

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
    ordered_short_labels = [e.replace(' adjustment coefficient', '') for e in ordered_labels]
    ax.bar(ordered_short_labels, percentage, color=tuple(ordered_colors))
    ax.set_xticklabels([])
    # ax.set_xticks(ordered_short_labels)
    # plt.setp(ax.get_xticklabels(), fontsize=5.5)
    ax.set_ylabel('Percentage of massifs (%)')
    ax.set_xlabel('Parameterization that minimizes the mean log score')

    plot("histo parametrization", folder, show)


def plot(filename, folder, show):
    if show:
        plt.show()
    else:
        filename = op.join(folder, filename)
        StudyVisualizer.savefig_in_results(filename, transparent=True)
        StudyVisualizer.savefig_in_results(filename, transparent=False)
    plt.close()


def massif_name_to_nb_linear_pieces(massif_names, min_mode, altitude, snowfall_str):
    massif_name_to_nb_linear_pieces = OrderedDict()
    print('start table with model as truth:')
    for massif_name in sorted(massif_names):
        print(massif_name.replace('_', ''), end=' & ')
        nb1 = _get_nb_linear_pieces(massif_name, model_as_truth_excel_folder.format(altitude, snowfall_str))
        if min_mode:
            nb2 = _get_nb_linear_pieces(massif_name, calibration_aic_excel_folder.format(altitude, snowfall_str))
            updated_nb = min(nb1, nb2)
        else:
            updated_nb = nb1
        massif_name_to_nb_linear_pieces[massif_name] = updated_nb
    print('end table with model as truth')
    return massif_name_to_nb_linear_pieces


def _get_nb_linear_pieces(massif_name, excel_folder, ):
    s = load_serie_parametrization_without_adjustement(massif_name, numbers_of_pieces, excel_folder)
    best_idx = s.idxmin()
    l = [str(round(v, 2)) for v in s.values]
    print(" & ".join(["\\textbf{" + number + '}' if idx == best_idx else number for idx, number in zip(s.index, l)]),
          '\\\\')
    return best_idx


def load_serie_parametrization_without_adjustement(massif_name, numbers_of_pieces, excel_folder):
    df = load_df_complete(massif_name, numbers_of_pieces, excel_folder)
    return df.iloc[0]


def load_df_complete(massif_name, numbers_of_pieces, excel_folder):
    df = pd.DataFrame()
    for number_of_pieces in numbers_of_pieces:
        s = _load_dataframe(massif_name, number_of_pieces, excel_folder)
        df = pd.concat([df, s], axis=1)
    df = df.iloc[[2 * i + 1 for i in range(len(df) // 2)]]
    df.columns = numbers_of_pieces
    return df


def _load_dataframe(massif_name, number_of_pieces, excel_folder):
    model_name = number_to_model_name[number_of_pieces]
    short_excel_folder = excel_folder.split('/')[-2:]
    assert op.exists(excel_folder), short_excel_folder
    files = [f for f in os.listdir(excel_folder) if model_name in f]
    assert len(files) == 1, "{} {}".format(short_excel_folder, model_name, files)
    filepath = op.join(excel_folder, files[0])
    df = pd.read_excel(filepath)
    columns = [c for c in df.columns if massif_name in c]
    s = df.loc[:, columns].mean(axis=1)
    if len(df) == 12:
        s = s.iloc[2:]
    df = pd.DataFrame.from_dict({number_of_pieces: s})
    return df


if __name__ == '__main__':
    # massif_name_to_nb_linear_pieces(AbstractStudy.all_massif_names())
    massif_names = AbstractStudy.all_massif_names()
    # massif_names = ["Vanoise"]

    run_selection(massif_names, 1500, show=False, snowfall=False)
    # run_selection(massif_names, 900, show=True, snowfall=False)
    # run_selection(massif_names, 2100, show=True, snowfall=False)

    # massif_name_to_nb_linear_pieces_and_parametrization(['Vanoise'])
