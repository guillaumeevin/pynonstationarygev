
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

excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/abstract_experiments/CalibrationValidationExperiment"
excel_start = 'AdamontPrecipitation_1500m_4couples_testTrue_NonStationaryLocationAndScaleAndShapeTemporalModel'
excel_start = 'AdamontPrecipitation_1500m_20couples_testFalse_NonStationaryLocationAndScaleAndShapeTemporalModel'
# excel_start = 'AdamontPrecipitation_1500m_20couples_testTrue_NonStationaryLocationAndScaleAndShapeTemporalModel'

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
        label = None if is_train else short_name[5:]
        # color = short_name_to_color[short_name]
        color = i_to_color[i]
        mean_scores = [year_to_name_to_mean_score[year][name] for year in years]
        ax.plot(percentages, mean_scores, label=label, linestyle=linestyle, color=color)
    ax.legend()
    ax.set_xlabel('Percentages of year for the calibration of the split-sample experiment (\%)')
    ax.set_ylabel('Mean logarithmic score')
    plt.show()

short_name_to_color = {
    "no effect": "blue"
    "loc_gcm scale_gcm"
}

i_to_color = {
    0: 'blue',
    4: 'violet',
    2: 'red',
    1: 'orange',
    5: 'yellow',
}




if __name__ == '__main__':
    main()

