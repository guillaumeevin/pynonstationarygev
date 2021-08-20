from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from projects.projected_extreme_snowfall.results.experiment.abstract_experiment import AbstractExperiment
from projects.projected_extreme_snowfall.results.experiment.calibration_validation_experiment import \
    CalibrationValidationExperiment
from root_utils import VERSION_TIME

excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/v2"
# excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv"
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/sensitivity_1500"
# excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/sensitivity_900"
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/sensitivity_1500_gcm_2019"
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/sensitivity_900_gcm_2019"
# excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/sensitivity_2100_gcm_2019"
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/snowfall_sensitivity_1500_2019"
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/precipitation_sensitivity_1500_2019"
# excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/precipitation_sensitivity_1500_None"

import os.path as op
import os
import pandas as pd
import matplotlib.pyplot as plt

excel_start = 'AdamontSnowLoad_1500m_20couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_'
# excel_start = 'AdamontSnowLoad_2100m_20couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_'
excel_start = "AdamontSnowLoad_1500m_1couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1"
excel_start = "AdamontSnowLoad_2100m_1couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1"
excel_start = "AdamontSnowLoad_2100m_1couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w4_"
excel_start = "AdamontSnowLoad_2100m_1couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w40"
excel_start = "AdamontSnowLoad_2100m_1couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w2"
excel_start = "AdamontSnowLoad_2100m_1couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w10"
excel_start = "AdamontSnowLoad_1500m_1couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1_"
excel_start = "AdamontSnowLoad_2100m_1couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1_"
excel_start = "AdamontSnowLoad_1500m_20couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1_True_"

excel_start = "AdamontSnowLoad_1500m_20couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1_(True"


# excel_start = "AdamontSnowLoad_1500m_20couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1_(False"
#


def main(massif_name, folder_idx=0):
    folder = ["CalibrationValidationExperiment", "CalibrationValidationExperimentPast", "ModelAsTruthExperiment"][
        folder_idx]
    if folder_idx in [0, 1]:
        excel_start = "AdamontSnowLoad_1500m_20couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1_(False, False, False)_None"
        excel_start = "AdamontSnowLoad_1500m_20couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1_(False, False, False)_2019"
        excel_start = "AdamontSnowLoad_1500m_20couples_testFalse_NonStationaryLocationAndScaleTemporalModel_w1_(False, False, False)_None"
        # excel_start = "AdamontSnowLoad_1500m_20couples_testFalse_NonStationaryLocationAndScaleGumbelModel_w1_(False, False, False)_None"
        excel_start = "AdamontSnowLoad_1500m_20couples_testFalse_NonStationaryLocationAndScaleGumbelModel_w1_(False, True, False)_2019"
        excel_start = "AdamontSnowLoad_1500m_20couples_testFalse_NonStationaryLocationAndScaleGumbelModel_w1_(False, False, False)_None"
        # excel_start = "AdamontSnowLoad_1500m_20couples_testFalse_NonStationaryLocationAndScaleGumbelModel_w1_(True, False, False)_None"
        # for i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20][-4:]:
        #     excel_start = "AdamontSnowLoad_1500m_{}couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1_(False, False, False)".format(i)
        plot(excel_start, folder, folder_idx, massif_name)

    else:
        excel_start = "AdamontSnowLoad_1500m_3couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1_(False, False, False)_2019"
        plot(excel_start, folder, folder_idx, massif_name)


def plot(excel_start, folder, folder_idx, massif_name, nb_gcm=1):
    excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments"
    excel_path = op.join(excel_path, folder)
    year_to_name_to_mean_score = {}
    for f in os.listdir(excel_path):
        if f.startswith(excel_start) and f.endswith("xlsx"):
            filepath = op.join(excel_path, f)
            df = pd.read_excel(filepath)
            year = f.split('_')[-1].split('.')[0]
            df2 = df.iloc[:, :-1]
            ind = ~df2.isnull().any(axis=0)
            if massif_name is not None:
                ind = ind & df2.columns.str.contains(massif_name)
            d = {}
            for i in range(len(df)):
                k = df2.index[i]
                row = df2.iloc[i, :]
                value = row.loc[ind].mean()
                d[k] = value
            year_to_name_to_mean_score[year] = d
    for year, d in year_to_name_to_mean_score.items():
        for name, score in d.items():
            print(name, score)
            # year_to_name_to_mean_score[year] = {df.index[i]: df.loc[ind].mean() for i in range(len(df))}
    # Plot
    years = sorted(list(year_to_name_to_mean_score.keys()))
    print(years)
    percentages = [round(100 * (int(year) + 1 - 1959) / 61, 2) for year in years]
    percentages = [round(p / 10) * 10 for p in percentages]
    # Select only some of them
    # years = years[-4:]
    # percentages = percentages[-4:]
    if folder_idx == 1:
        percentages = percentages[::-1]
    names = list(df.index)
    ax = plt.gca()
    prefix_found = set()
    for j, name in enumerate(names):
        prefix, *rest = name.split('_')
        prefix_found.add(prefix)
        linestyle = prefix_to_linestyle[prefix]
        short_name = '_'.join(rest)
        if "with obs and " in short_name:
            short_name = short_name.replace("with obs and ", "")
        if '_' in short_name:
            short_name = '_'.join(short_name.split()[0].split('_')[1:])
        label = short_name_to_label[short_name] if prefix == 'ValidationObs' else None
        color = short_name_to_color[short_name]
        mean_scores = [year_to_name_to_mean_score[year][name] for year in years]
        if "scale" in '_'.join(rest):
            marker = "x"
            if label is not None:
                label += 'with scale'
        else:
            marker = None
        ax.plot(percentages, mean_scores, label=label, linestyle=linestyle, color=color, marker=marker)
    ax.legend(loc='upper left')
    ax3 = ax.twiny()
    ax3.set_xticks(percentages)
    ax3.set_xlim(ax.get_xlim())
    ax3.set_xticklabels(['{}'.format(year) for year in years])
    ax3.set_xlabel('Last year where observations are available for the calibration')
    ax.set_xticks(percentages)
    ax.set_xticklabels(['{}%'.format(p) for p in percentages])
    ax.set_xlabel('Percentage of observation for the calibration (%)')
    ax.set_ylabel('Mean logarithmic score')
    ax2 = ax.twinx()
    legend_elements = [
        Line2D([0], [0], color='k', lw=1, label=prefix_to_label[prefix],
               linestyle=prefix_to_linestyle[prefix])
        for prefix in prefix_found
    ]
    ax2.legend(handles=legend_elements, loc='lower left')
    ax2.set_yticks([])
    a,b= ax.get_ylim()
    ax.set_ylim(0.85*a, b*1.25)
    filename = "all" if massif_name is None else massif_name
    filename = op.join(VERSION_TIME, folder, filename + str(nb_gcm))
    StudyVisualizer.savefig_in_results(filename, transparent=False)
    plt.close()


linestyles = ['--', '-', 'dotted', 'dashdot']
prefix_to_linestyle = dict(zip(AbstractExperiment.prefixs, linestyles))
labels = ["Score for obs on calibration set", "Score for obs on validation set",
          "Score for ensemble members on calibration period", "Score on the complete calibration set"]
prefix_to_label = dict(zip(AbstractExperiment.prefixs, labels))

short_name_to_color = {
    "without obs": "grey",
    "no effect": "blue",
    "is_ensemble_member": 'yellow',
    "gcm": 'orange',
    "rcm": "red",
    "gcm_and_rcm": 'violet',
}

short_name_to_label = {
    'without obs': "Baseline",
    "no effect": "Zero adjustment coefficients",
    "gcm": 'One adjustment coefficient for each GCM',
    "gcm_and_rcm": 'One adjustment coefficient for each GCM-RCM pair',
    "is_ensemble_member": 'One adjustment coefficient for all GCM-RCM pairs',
    "rcm": "One adjustment coefficient for each RCM",
}

if __name__ == '__main__':
    for folder_idx in [0, 2][:1]:
        for massif_name in [None] + AbstractStudy.all_massif_names():
            main(massif_name, folder_idx)
