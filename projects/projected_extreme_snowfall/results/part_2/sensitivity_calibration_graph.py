
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/v2"
# excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv"
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/sensitivity_1500"
# excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/sensitivity_900"
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/sensitivity_1500_gcm_2019"
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/sensitivity_900_gcm_2019"
# excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/sensitivity_2100_gcm_2019"

import os.path as op
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    year_to_name_to_mean_score = {}
    for f in os.listdir(excel_path):
        if f.endswith("xlsx"):
            print("\n", f)
            filepath = op.join(excel_path, f)
            df = pd.read_excel(filepath)
            year = f.split('splityear_')[-1][:4]
            year_to_name_to_mean_score[year] = {df.index[i]: -df.iloc[i, :-1].mean() for i in range(5)}
    # Plot
    years = sorted(list(year_to_name_to_mean_score.keys()))
    percentages = [100 * (int(year) - 1959) / 61 for year in years]
    names = list(df.index)
    ax = plt.gca()
    for name in names:
        mean_scores = [year_to_name_to_mean_score[year][name] for year in years]
        ax.plot(percentages, mean_scores, label=name)
    ax.legend()
    ax.set_xlabel('Percentages of year for the calibration of the split-sample experiment (\%)')
    ax.set_ylabel('Mean logarithmic score')
    plt.show()





if __name__ == '__main__':
    main()

