from matplotlib.lines import Line2D

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

excel_start = 'AdamontPrecipitation_1500m_20couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel_'

folder = ["CalibrationValidationExperiment", "ModelAsTruthExperiment"][0]
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments"
excel_path = op.join(excel_path, folder)

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
    percentages = [100 * (int(year) - 1959) / 61 for year in years]
    names = list(df.index)
    ax = plt.gca()
    for j, name in enumerate(names):
        i = j // 2 if j <= 5 else 1 + j // 2
        is_train = 'Train' in name
        linestyle = '--' if is_train else '-'
        short_name = '_'.join(name.split('_')[1:])
        label = None if is_train else short_name_to_label[short_name]
        color = short_name_to_color[short_name]
        mean_scores = [year_to_name_to_mean_score[year][name] for year in years]
        ax.plot(percentages, mean_scores, label=label, linestyle=linestyle, color=color)
    ax.legend()
    ax.set_xticks(percentages)
    ax.set_xticklabels(['1959-{}'.format(year) for year in years])
    ax.set_xlabel('Period where the observations are available for the calibration')
    ax.set_ylabel('Mean logarithmic score')
    ax2 = ax.twinx()
    legend_elements = [
        Line2D([0], [0], color='k', lw=1, label="Score on calibration set",
               linestyle="--"),
        Line2D([0], [0], color='k', lw=1, label="Score on validation set",
               linestyle="-")
    ]
    ax2.legend(handles=legend_elements, loc='center right')
    plt.show()

short_name_to_color = {
    "no effect": "blue",
    "loc_gcm scale_gcm": 'orange',
    "loc_gcm_and_rcm scale_gcm_and_rcm": 'violet',
    "loc_is_ensemble_member scale_is_ensemble_member": 'yellow',
    "loc_rcm scale_rcm": "red",
}

short_name_to_label = {
    "no effect": "Zero adjustment coefficients",
    "loc_gcm scale_gcm": 'One adjustment coefficient for each GCM',
    "loc_gcm_and_rcm scale_gcm_and_rcm": 'One adjustment coefficient for each GCM-RCM pair',
    "loc_is_ensemble_member scale_is_ensemble_member": 'One adjustment coefficient for all GCM-RCM pairs',
    "loc_rcm scale_rcm": "One adjustment coefficient for each RCM",
}




if __name__ == '__main__':
    main()

