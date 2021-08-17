from matplotlib.lines import Line2D

from projects.projected_extreme_snowfall.results.experiment.abstract_experiment import AbstractExperiment
from projects.projected_extreme_snowfall.results.experiment.calibration_validation_experiment import \
    CalibrationValidationExperiment

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
# excel_start = "AdamontPrecipitation_1500m_20couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_w1_True_"
#
folder = ["CalibrationValidationExperiment", "ModelAsTruthExperiment"][0]
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments"
excel_path = op.join(excel_path, folder)


class AbstractSensitivityExperimentGraph(object):

    def plot(self):
        pass


def main():
    year_to_name_to_mean_score = {}
    for f in os.listdir(excel_path):
        if f.startswith(excel_start) and f.endswith("xlsx"):
            print("\n", f)
            filepath = op.join(excel_path, f)
            df = pd.read_excel(filepath)
            year = f.split('_')[-2]
            df2 = df.iloc[:, :-1]
            ind = ~df2.isnull().any(axis=0)
            d = {}
            for i in range(len(df)):
                k = df2.index[i]
                row = df2.iloc[i, :]
                value = row.loc[ind].mean()
                d[k] = value
            year_to_name_to_mean_score[year] = d
    for year, d in year_to_name_to_mean_score.items():
        print(year)
        for name, score in d.items():
            print(name, score)
            # year_to_name_to_mean_score[year] = {df.index[i]: df.loc[ind].mean() for i in range(len(df))}
    # Plot
    years = sorted(list(year_to_name_to_mean_score.keys()))
    percentages = [round(100 * (int(year) + 1 - 1959) / 61, 2) for year in years]
    # percentages = [round(p / 10) * 10 for p in percentages]

    names = list(df.index)
    ax = plt.gca()
    prefix_found = set()
    for j, name in enumerate(names):
        prefix, *rest = name.split('_')
        prefix_found.add(prefix)
        linestyle = prefix_to_linestyle[prefix]
        short_name = '_'.join(rest)
        label = short_name_to_label[short_name] if prefix == 'ValidationObs' else None
        color = short_name_to_color[short_name]
        mean_scores = [year_to_name_to_mean_score[year][name] for year in years]
        ax.plot(percentages, mean_scores, label=label, linestyle=linestyle, color=color)
    ax.legend()
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
    ax2.legend(handles=legend_elements)
    ax2.set_yticks([])
    plt.show()


linestyles = ['--', '-', 'dotted', 'dashdot']
prefix_to_linestyle = dict(zip(AbstractExperiment.prefixs, linestyles))
labels = ["Score for obs on calibration set", "Score for obs on validation set",
          "Score for ensemble members on calibration period", "Score on the complete calibration set"]
prefix_to_label = dict(zip(AbstractExperiment.prefixs, labels))

short_name_to_color = {
    "no effect": "blue",
    "loc_gcm scale_gcm": 'orange',
    "loc_gcm_and_rcm scale_gcm_and_rcm": 'violet',
    "loc_is_ensemble_member scale_is_ensemble_member": 'yellow',
'loc_is_ensemble_member scale_is_ensemble_member shape_is_ensemble_member': 'green',
    "loc_rcm scale_rcm": "red",
'loc_is_ensemble_member': "grey",
'scale_is_ensemble_member': 'violet'
}

short_name_to_label = {
    "no effect": "Zero adjustment coefficients",
    "loc_gcm scale_gcm": 'One adjustment coefficient for each GCM',
    "loc_gcm_and_rcm scale_gcm_and_rcm": 'One adjustment coefficient for each GCM-RCM pair',
    "loc_is_ensemble_member scale_is_ensemble_member": 'One adjustment coefficient for all GCM-RCM pairs',
    "loc_rcm scale_rcm": "One adjustment coefficient for each RCM",
'loc_is_ensemble_member scale_is_ensemble_member shape_is_ensemble_member': 'is ensemble with shape',
'loc_is_ensemble_member': 'is ensemble loc only',
'scale_is_ensemble_member': 'is ensemble scale only'
}

if __name__ == '__main__':
    main()
