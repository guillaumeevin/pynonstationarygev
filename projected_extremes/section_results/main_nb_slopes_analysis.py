from collections import Counter

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from projected_extremes.section_results.utils.selection_utils import short_name_to_color
from root_utils import VERSION_TIME

model_as_truth_excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/ModelAsTruthExperiment/2100 snowfall"
# model_as_truth_excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/CalibrationAicExperiment/2100 snowfall"

split_sample_excel_filepath = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/CalibrationValidationExperiment/AdamontSnowfall_2100m_20couples_testFalse_NonStationaryFourLinearLocationAndScaleAndShapeModel_w1_(False, False, False)_None_1988.xlsx"
# split_sample_excel_filepath = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/CalibrationValidationExperiment/AdamontSnowfall_2100m_20couples_testFalse_NonStationaryFourLinearLocationAndScaleAndShapeModel_w1_(False, False, False)_None_1995.xlsx"
# split_sample_excel_filepath = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/CalibrationValidationExperiment/AdamontSnowfall_2100m_20couples_testFalse_NonStationaryFourLinearLocationAndScaleAndShapeModel_w1_(False, False, False)_None_2001.xlsx"
# split_sample_excel_filepath = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/CalibrationValidationExperiment/AdamontSnowfall_2100m_20couples_testFalse_NonStationaryFourLinearLocationAndScaleAndShapeModel_w1_(False, False, False)_None_2007.xlsx"
# split_sample_excel_filepath = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/CalibrationValidationExperiment/AdamontSnowfall_2100m_20couples_testFalse_NonStationaryFourLinearLocationAndScaleAndShapeModel_w1_(False, False, False)_None_2013.xlsx"

import os.path as op
import os
import pandas as pd
import matplotlib.pyplot as plt






def main_map_optimal_number_of_pieces(massif_names):
    ax = plt.gca()
    numbers_of_pieces = list(range(1, 7))
    massif_name_to_optimal_short_name_and_number_pieces = load_massif_name_to_optimal_combination(massif_names, numbers_of_pieces)
    massif_name_to_text = {m: str(n) for m, (_, n) in massif_name_to_optimal_short_name_and_number_pieces.items()}
    massif_name_to_color = {m: short_name_to_color[s] for m, (s, _) in massif_name_to_optimal_short_name_and_number_pieces.items()}
    AbstractStudy.visualize_study(ax, massif_name_to_color=massif_name_to_color,
                                  massif_name_to_text=massif_name_to_text,
                                  add_text=True,
                                  axis_off=True)
    plt.show()


def main_map_optimal_parametrization(massif_names):
    ax = plt.gca()
    df = pd.read_excel(split_sample_excel_filepath).iloc[2:]
    df = df.iloc[[2 * i + 1 for i in range(5)]]

    massif_name_to_color = {}
    for massif_name in massif_names:
        c_massif = [c for c in df.columns if massif_name in c]
        s = df[c_massif]
        index = s.idxmin().values[0]
        optimal_short_name = get_short_name(index)
        color = short_name_to_color[optimal_short_name]
        massif_name_to_color[massif_name] = color
    massif_name_to_text = {m: m.replace('_', '-') for m in massif_names}
    AbstractStudy.visualize_study(ax, massif_name_to_color=massif_name_to_color,
                                  massif_name_to_text=massif_name_to_text,
                                  add_text=True,
                                  axis_off=True)
    plt.show()


def main_histogram_optimal_number_pieces(massif_names):
    numbers_of_pieces = list(range(0, 5))
    massif_name_to_optimal_short_name_and_number_pieces = load_massif_name_to_optimal_combination(massif_names, numbers_of_pieces)
    numbers = [number for massif_name, (_, number)
               in massif_name_to_optimal_short_name_and_number_pieces.items()]
    c = Counter(numbers)
    number_of_occurences = [c[n] if n in c else 0 for n in numbers_of_pieces]
    percentage = [100 * n / sum(number_of_occurences) for n in number_of_occurences]
    ax = plt.gca()
    ax.bar(numbers_of_pieces, percentage)
    ax.set_xticks(numbers_of_pieces)
    ax.set_ylabel('Percentage of massifs (%)')
    ax.set_xlabel('Number of linear pieces that minimizes the mean log score')
    plt.show()


def main_histogram_optimal_parametrization(massif_names):
    pass


def load_massif_name_to_optimal_combination(massif_names, numbers_of_pieces):
    massif_name_to_optimal_short_name_and_number_pieces = {}
    for massif_name in massif_names:
        df = load_df_complete(massif_name, numbers_of_pieces)
        l = []
        for i, row in df.iterrows():
            min, argmin = row.min(), row.idxmin()
            l.append((i, argmin, min))
        t_optimal = sorted(l, key=lambda t: t[2])[0]
        short_name = get_short_name(t_optimal[0])
        massif_name_to_optimal_short_name_and_number_pieces[massif_name] = (short_name, t_optimal[1])
    return massif_name_to_optimal_short_name_and_number_pieces


def main_plot_optimal_model(massif_name):
    ax = plt.gca()
    numbers_of_pieces = list(range(1, 7))
    df = load_df_complete(massif_name, numbers_of_pieces)
    # print("Mean", massif_name, df.transpose().mean(axis=1).idxmin())
    # print("Min", massif_name, df.transpose().min(axis=1).idxmin())
    # df = df.transpose()
    # df["CalibrationObs_with obs and mean"] = df.mean(axis=1)
    # df = df.transpose()

    for i, row in df.iterrows():
        short_name = get_short_name(i)
        linestyle = None
        label = short_name_to_label[short_name]
        color = short_name_to_color[short_name]
        ax.plot(list(row.index), list(row.values),
                color=color, label=label, linestyle=linestyle)
    ax.set_xticks(numbers_of_pieces)
    ax.set_xlabel('Number of linear pieces')
    ax.set_ylim(top=ax.get_ylim()[1] * 1.002)
    ylabel = 'Mean log score'
    if massif_name is not "mean":
        ylabel += " for the {}".format(massif_name.replace("_", "-"))
    ax.set_ylabel(ylabel)
    ax.legend()
    filename = op.join(VERSION_TIME, "optimal_nb_slopes", ylabel)
    StudyVisualizer.savefig_in_results(filename, transparent=False)
    plt.close()


if __name__ == '__main__':
    massif_names = ['mean']
    massif_names += AbstractStudy.all_massif_names()[:]
    for massif_name in massif_names:
        print(massif_name)
        main_plot_optimal_model(massif_name)

    # main_plot_optimal_model('mean')
    # main_plot_optimal_model('Devoluy')
    main_map_optimal_number_of_pieces(AbstractStudy.all_massif_names()[:])
    # main_histogram_optimal_number_pieces(AbstractStudy.all_massif_names()[:])


    # main_map_optimal_parametrization(AbstractStudy.all_massif_names()[:])
