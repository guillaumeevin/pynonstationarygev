from collections import OrderedDict

import numpy as np
import pandas as pd

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy

DISTANCE = "Absolute distance to the mean bias of obs"

MEAN_BIAS = "Mean bias"

OBS = 'obs'


def ordered_gcm_rcm_couples_in_terms_of_bias_similar_to_bias_of_obs(nb_gcm_rcm_couples_as_truth, altitude, gcm_rcm_couples,
                                                                    massif_names, scenario, year_min, year_max,
                                                                    study_class, safran_study_class):

    assert len(massif_names) == 1
    massif_name = massif_names[0]
    # Load the study
    safran_study = safran_study_class(altitude=altitude, year_min=year_min, year_max=year_max)
    gcm_rcm_couple_to_study = OrderedDict()
    for gcm_rcm_couple in gcm_rcm_couples:
        study = study_class(altitude=altitude, scenario=scenario, gcm_rcm_couple=gcm_rcm_couple,
                                              year_min=year_min, year_max=year_max)
        gcm_rcm_couple_to_study[gcm_rcm_couple] = study
    # Load the obs columns
    df = pd.DataFrame({OBS: [compute_mean_bias_for_annual_maxima(massif_name, safran_study, study)
                             for study in gcm_rcm_couple_to_study.values()]},
                      index=gcm_rcm_couples)
    # Load the gcm_rcm columns
    for gcm_rcm_couple in gcm_rcm_couples:
        gcm, rcm = gcm_rcm_couple
        if (gcm == 'HadGEM2-ES') or (rcm == 'RCA4'):
            continue
        adamont_study = gcm_rcm_couple_to_study[gcm_rcm_couple]
        df[gcm_rcm_couple] = [compute_mean_bias_for_annual_maxima(massif_name, adamont_study, study) if c != gcm_rcm_couple else np.nan
                               for c, study in gcm_rcm_couple_to_study.items()]
    # Transpose
    df = df.transpose()
    # Compute the mean
    df[MEAN_BIAS] = df.mean(axis=1)
    # COmpute
    df[DISTANCE] = (df[MEAN_BIAS] - df.loc[OBS, MEAN_BIAS]).abs()
    df.sort_values(by=DISTANCE, inplace=True)


    # # Some prints for the presentation
    df_small = df.iloc[:9 + 1, -2:]
    df_small = df_small.round(decimals=2)
    # print(df_small)
    # df_small.to_csv("altitude={}.csv".format(altitude))

    print('\ndistance for altitude={}'.format(altitude))
    print(df.loc[:, [MEAN_BIAS, DISTANCE]].iloc[:nb_gcm_rcm_couples_as_truth+1])

    gcm_rcm_list = list(df.index)[1:nb_gcm_rcm_couples_as_truth + 1]
    assert len(gcm_rcm_list) == nb_gcm_rcm_couples_as_truth
    return gcm_rcm_list

def compute_mean_bias_for_annual_maxima(massif_name, study_reference: AbstractStudy, study_for_comparison: AbstractAdamontStudy):
    start_1, end_1 = study_reference.start_year_and_stop_year
    maxima1 = study_reference.massif_name_to_annual_maxima[massif_name]
    start_2, end_2 = study_for_comparison.start_year_and_stop_year
    maxima2 = study_for_comparison.massif_name_to_annual_maxima[massif_name]
    start = max(start_1, start_2)
    end = min(end_1, end_2)
    bias_list = [maxima2[year-start_2] - maxima1[year-start_1] for year in range(start, end+1)]
    return np.mean(bias_list)
