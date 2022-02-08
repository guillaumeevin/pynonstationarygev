
excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv/v2"
# excel_path = "/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/model_as_truth_csv"
import os.path as op
import os
import pandas as pd

def main():
    for f in os.listdir(excel_path):
        if f.endswith("xlsx") and "last" in f:
            print("\n", f)
            filepath = op.join(excel_path, f)
            df = pd.read_excel(filepath)
            no_effect_index = list(df.index).index('no effect')
            print(df.iloc[:, -1])
            # for i in range(5):
            #     if i != no_effect_index:
            #         without_coef = -df.iloc[no_effect_index, :-1]
            #         with_coef = -df.iloc[i, :-1]
            #         # score = 100 * (with_coef - without_coef) / without_coef
            #         print()
            #         print(with_coef, without_coef)
            #         score = with_coef - without_coef
            #         # relative score is not ideal because our score are close to zero
            #         mean_relative_score = score.mean()
            #         print(df.index[i], mean_relative_score)
            for i in range(5):
                s = -df.iloc[i, :-1]
                print(df.index[i], round(s.mean(), 2))



if __name__ == '__main__':
    main()

