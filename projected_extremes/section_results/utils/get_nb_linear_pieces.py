import os
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt
import os.path as op

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_data.utils import DATA_PATH, RESULTS_PATH
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from projected_extremes.section_results.utils.selection_utils import get_short_name, number_to_model_class, \
    short_name_to_parametrization_number, linear_effects_for_selection, number_to_model_name, short_name_to_color, \
    short_name_to_label
from projected_extremes.section_results.utils.setting_utils import get_last_year_for_the_train_set, set_up_and_load
from root_utils import VERSION_TIME

abstract_experiment_folder = op.join(RESULTS_PATH, "abstract_experiments")
model_as_truth_excel_folder = op.join(abstract_experiment_folder, "ModelAsTruthExperiment/{} {}")
calibration_excel_folder = op.join(abstract_experiment_folder, "CalibrationValidationExperiment/{}_{}_{}")


def eliminate_massif_name_with_too_much_zeros(massif_names, altitude, gcm_rcm_couples,
                                              safran_study_class, scenario,
                                              study_class, season, snowfall
                                              ):
    new_massif_names = []
    for massif_name in massif_names:

        gcm_rcm_couple_to_studies = VisualizerForProjectionEnsemble.load_gcm_rcm_couple_to_studies([altitude],
                                                                                                   gcm_rcm_couples,
                                                                                                   None,
                                                                                                   safran_study_class,
                                                                                                   scenario,
                                                                                                   season,
                                                                                                   study_class)
        safran_studies = gcm_rcm_couple_to_studies[(None, None)]
        if massif_name in safran_studies.study.study_massif_names:
            nb_zeros = 0
            nb_data = 0
            for studies in gcm_rcm_couple_to_studies.values():
                dataset = studies.spatio_temporal_dataset(massif_name=massif_name, massif_altitudes=[altitude])
                s = dataset.observations.df_maxima_gev.iloc[:, 0]
                nb_zeros += sum(s == 0)
                nb_data += len(s)
            if snowfall is True:
                threshold = 0.04
            else:
                threshold = 0.4

            threshold_massif = 100 * nb_zeros / nb_data
            if threshold_massif > threshold:
                print('eliminate due to zeros:', massif_name)
            else:
                new_massif_names.append(massif_name)
    return new_massif_names


def run_selection(massif_names, altitude, gcm_rcm_couples,
                  safran_study_class, scenario,
                  study_class, show=False,
                  snowfall=False,
                  season=Season.annual,
                  print_latex_table=False,
                  ):
    massif_name_to_number, linear_effects, massif_names, snowfall_str, numbers_of_pieces = get_massif_name_to_number(
        altitude, gcm_rcm_couples, massif_names,
        safran_study_class, scenario, snowfall,
        study_class, season)

    massif_name_to_short_name = {}
    if print_latex_table:
        print('\n\nstart table with split-sample:')
    print(massif_name_to_number)
    for massif_name, number in massif_name_to_number.items():
        if print_latex_table:
            print(massif_name.replace('_', ' '), end=' & ')
        percentages = [0.6, 0.7, 0.8][:]
        df = pd.DataFrame()
        for p in percentages:
            year = get_last_year_for_the_train_set(p)
            excel_folder = calibration_excel_folder.format(snowfall_str, altitude, year)
            df2 = load_df_complete(massif_name, [number], excel_folder, linear_effects, gcm_rcm_couples)
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
        if print_latex_table:
            print(
                " & ".join(
                    ["\\textbf{" + number + '}' if idx == best_idx else number for idx, number in zip(s2.index, l)]),
                '\\\\')
        if not isinstance(best_idx, str) and np.isnan(best_idx):
            massif_names.remove(massif_name)
            continue
        short_name = get_short_name(best_idx)
        massif_name_to_short_name[massif_name] = short_name
    if print_latex_table:
        print('\n\nend with split-sample:')

    plots(massif_name_to_short_name, massif_name_to_number, show, altitude, snowfall_str, numbers_of_pieces)

    massif_name_to_model_class = {m: number_to_model_class[n] for m, n in massif_name_to_number.items()}
    massif_name_to_parametrization_number = {m: short_name_to_parametrization_number[s] for m, s in
                                             massif_name_to_short_name.items()}
    return massif_names, massif_name_to_model_class, massif_name_to_parametrization_number, linear_effects_for_selection


def get_massif_name_to_number(altitude, gcm_rcm_couples, massif_names, safran_study_class, scenario, snowfall,
                              study_class, season):
    max_number_of_pieces, min_number_of_pieces, snowfall_str = get_min_max_number_of_pieces(snowfall)
    numbers_of_pieces = list(range(min_number_of_pieces, max_number_of_pieces + 1))
    massif_names = eliminate_massif_name_with_too_much_zeros(massif_names, altitude, gcm_rcm_couples,
                                                             safran_study_class, scenario,
                                                             study_class, season, snowfall)
    d = massif_name_to_nb_linear_pieces(massif_names, altitude,
                                        snowfall_str, numbers_of_pieces, gcm_rcm_couples)

    massif_names = list(d.keys())
    linear_effects = linear_effects_for_selection
    return d, linear_effects, massif_names, snowfall_str, numbers_of_pieces


def get_min_max_number_of_pieces(snowfall):
    if snowfall is True:
        snowfall_str = 'snowfall'
        min_number_of_pieces = 1
        max_number_of_pieces = 4
    elif snowfall is None:
        snowfall_str = 'precipitation'
        min_number_of_pieces = 1
        max_number_of_pieces = 4
    else:
        min_number_of_pieces = 1
        max_number_of_pieces = 4
        snowfall_str = "snow load"
    return max_number_of_pieces, min_number_of_pieces, snowfall_str


def plots(massif_name_to_short_name, d, show, altitude, snowfall_str, numbers_of_pieces):
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
    plot("legend", folder, show, altitude, snowfall_str)

    # Plot the map
    ax = plt.gca()
    massif_name_to_color = {m: short_name_to_color[s] for m, s in massif_name_to_short_name.items()}

    massif_name_to_hatch_boolean_list = {}
    for massif_name in set(AbstractStudy.all_massif_names()) - set(list(massif_name_to_color.keys())):
        massif_name_to_hatch_boolean_list[massif_name] = [True, True]

    AbstractStudy.visualize_study(ax, massif_name_to_color=massif_name_to_color,
                                  massif_name_to_hatch_boolean_list=massif_name_to_hatch_boolean_list,
                                  massif_name_to_text={m: str(v) for m, v in d.items()},
                                  add_text=True,
                                  axis_off=True, show=False)

    plot("map", folder, show, altitude, snowfall_str)

    # plot the histogram for the number of pieces
    c = Counter(d.values())
    number_of_occurences = [c[n] if n in c else 0 for n in numbers_of_pieces]
    if sum(number_of_occurences) > 0:
        percentage = [100 * n / sum(number_of_occurences) for n in number_of_occurences]
        ax = plt.gca()
        ax.bar(numbers_of_pieces, percentage)
        ax.set_xticks(numbers_of_pieces)
        ax.set_ylabel('Percentage of massifs (%)')
        ax.set_xlabel('Number of linear pieces that minimizes the mean log score')

        plot("histo number pieces", folder, show, altitude, snowfall_str)

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

    plot("histo parametrization", folder, show, altitude, snowfall_str)


def plot(filename, folder, show, altitude, snowfall_str):
    if show:
        plt.show()
    else:
        filename = op.join(folder, filename + 'for {} at {}m'.format(snowfall_str, altitude))
        StudyVisualizer.savefig_in_results(filename, transparent=True)
        StudyVisualizer.savefig_in_results(filename, transparent=False)
    plt.close()


def massif_name_to_nb_linear_pieces(massif_names, altitude, snowfall_str, numbers_of_pieces, gcm_rcm_couples,
                                    print_latex_table=False):
    massif_name_to_nb_linear_pieces = OrderedDict()
    if print_latex_table:
        print('start table with model as truth:')
    for massif_name in sorted(massif_names):
        if print_latex_table:
            print(massif_name.replace('_', ''), end=' & ')
        nb = _get_nb_linear_pieces(massif_name, model_as_truth_excel_folder.format(altitude, snowfall_str),
                                   numbers_of_pieces, gcm_rcm_couples, print_latex_table)
        if not np.isnan(nb):
            massif_name_to_nb_linear_pieces[massif_name] = nb
    if print_latex_table:
        print('end table with model as truth')
    return massif_name_to_nb_linear_pieces


def _get_nb_linear_pieces(massif_name, excel_folder, numbers_of_pieces, gcm_rcm_couples, print_latex_table=False):
    s = load_serie_parametrization_without_adjustement(massif_name, numbers_of_pieces, excel_folder, gcm_rcm_couples)
    best_idx = s.idxmin()
    l = [str(round(v, 2)) for v in s.values]
    if print_latex_table:
        print(
            " & ".join(["\\textbf{" + number + '}' if idx == best_idx else number for idx, number in zip(s.index, l)]),
            '\\\\')
    return best_idx


def load_serie_parametrization_without_adjustement(massif_name, numbers_of_pieces, excel_folder, gcm_rcm_couples):
    df = load_df_complete(massif_name, numbers_of_pieces, excel_folder, (False, False, False), gcm_rcm_couples)
    return df.iloc[0]


def load_df_complete(massif_name, numbers_of_pieces, excel_folder, linear_effects, gcm_rcm_couples):
    df = pd.DataFrame()
    for number_of_pieces in numbers_of_pieces:
        s = _load_dataframe(massif_name, number_of_pieces, excel_folder, linear_effects, gcm_rcm_couples)
        # print(, number_of_pieces, s)
        df = pd.concat([df, s], axis=1)
    df = df.iloc[[2 * i + 1 for i in range(len(df) // 2)]]
    df.columns = numbers_of_pieces
    return df


def _load_dataframe(massif_name, number_of_pieces, excel_folder, linear_effects, gcm_rcm_couples):
    model_name = number_to_model_name[number_of_pieces]
    short_excel_folder = excel_folder.split('/')[-2:]
    assert op.exists(excel_folder), "Run a {} for {}".format(*short_excel_folder)
    linear_effects_name = str(linear_effects)
    couplename = str(len(gcm_rcm_couples)) + 'couples'
    files = [f for f in os.listdir(excel_folder) if
             (model_name in f) and (linear_effects_name in f) and (couplename in f)]
    join_str = ' '.join([model_name] + [str(e) for e in linear_effects])
    assert len(files) > 0, "Run a {} for {}".format(short_excel_folder[0], join_str)
    assert len(files) < 2, "Too many files that correspond: {}".format(*files)
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
    massif_names = ["Bauges"]

    snowfall = True
    altitude_to_number_of_pieces = OrderedDict()
    for altitude in [1200, 1500, 1800, 2100, 2400][-2:]:
        _, gcm_rcm_couples, _, _, scenario, study_class, _, _, _, safran_study_class, _, season = set_up_and_load(False,
                                                                                                                  snowfall)

        # _, _, d, _ = run_selection(massif_names, altitude, gcm_rcm_couples, safran_study_class, scenario, study_class,
        #               show=False, snowfall=snowfall)

        d, *_ = get_massif_name_to_number(altitude, gcm_rcm_couples, massif_names, safran_study_class,
                                          scenario, snowfall, study_class, season)

        altitude_to_number_of_pieces[altitude] = d[massif_names[0]]
        # run_selection(massif_names, 900, show=True, snowfall=False)
        # run_selection(massif_names, 2100, show=True, snowfall=False)

        # massif_name_to_nb_linear_pieces_and_parametrization(['Vanoise'])
    print('\n\n')
    print(altitude_to_number_of_pieces)
