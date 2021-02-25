import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from extreme_trend.trend_test.visualizers import StudyVisualizerForNonStationaryTrends


def mix_dsitrbution_impact():
    altitudes = [300, 600, 900]
    altitude_to_couple = {altitude: [None, None] for altitude in altitudes}
    for j, fit_gev_only_on_non_null_maxima in enumerate([True, False]):
        for altitude in altitudes:
            altitude_to_couple[altitude][j] = StudyVisualizerForNonStationaryTrends(
                CrocusSnowLoadTotal(altitude=altitude),
                select_only_acceptable_shape_parameter=True,
                multiprocessing=True,
                save_to_file=True,
                fit_gev_only_on_non_null_maxima=fit_gev_only_on_non_null_maxima,
                fit_only_time_series_with_ninety_percent_of_non_null_values=True,
                show=False)
    pd.options.display.width = 0
    # Compare return level
    for altitude, (viz1, viz2) in altitude_to_couple.items():
        # for m in viz1.all_massif_name_to_eurocode_uncertainty_for_minimized_aic_model_class:
        m_to_eurocode_return_level_1 = viz1.all_massif_name_to_eurocode_uncertainty_for_minimized_aic_model_class()
        m_to_eurocode_return_level_2 = viz2.all_massif_name_to_eurocode_uncertainty_for_minimized_aic_model_class()

        s_diff = []
        massifs_names = set(viz1.massifs_names_with_year_without_snow).intersection(m_to_eurocode_return_level_1.keys())
        for m in massifs_names:
            return_level1 = m_to_eurocode_return_level_1[m].mean_estimate
            return_level2 = m_to_eurocode_return_level_2[m].mean_estimate
            difference = return_level1 - return_level2
            abs_difference = abs(difference)
            s_diff.append({'massif': m, 'abs(r1 - r2)': abs_difference, 'abs(r1 - r2)/r1': 100 * abs_difference/return_level1,
                           'abs(r1 - r2)/r2': 100 * abs_difference / return_level2,
                           'r1 (with mixed)': return_level1, 'r2 (without)': return_level2
                           })
        df = pd.DataFrame(s_diff)
        print(df)
        print(altitude, '\n', df.describe())

"""

   abs(r1 - r2)  abs(r1 - r2)/r1  abs(r1 - r2)/r2       massif  r1 (with mixed)  r2 (without)
0      0.000000         0.000000         0.000000   Belledonne         0.370008      0.370008
1      0.000564         0.186687         0.186340       Bauges         0.302025      0.302589
2      0.000000         0.000000         0.000000   Mont-Blanc         0.407016      0.407016
3      0.017822         2.427636         2.370098       Aravis         0.734127      0.751949
4      0.025847         7.197779         6.714485     Chablais         0.359097      0.384944
5      0.060221         9.472780         8.653092      Vanoise         0.635731      0.695952
6      0.099971        21.315869        17.570553       Oisans         0.468997      0.568967
7      0.004269         1.617252         1.643837  Beaufortain         0.263942      0.259673
8      0.000731         0.206005         0.205581    Maurienne         0.354998      0.355729
300 
        abs(r1 - r2)  abs(r1 - r2)/r1  abs(r1 - r2)/r2  r1 (with mixed)  r2 (without)
count      9.000000         9.000000         9.000000         9.000000      9.000000
mean       0.023269         4.713779         4.149332         0.432882      0.455203
std        0.034915         7.110878         5.938525         0.156124      0.174963
min        0.000000         0.000000         0.000000         0.263942      0.259673
25%        0.000564         0.186687         0.186340         0.354998      0.355729
50%        0.004269         1.617252         1.643837         0.370008      0.384944
75%        0.025847         7.197779         6.714485         0.468997      0.568967
max        0.099971        21.315869        17.570553         0.734127      0.751949
   abs(r1 - r2)  abs(r1 - r2)/r1  abs(r1 - r2)/r2                massif  r1 (with mixed)  r2 (without)
0           0.0              0.0              0.0  Haut_Var-Haut_Verdon         1.977354      1.977354
1           0.0              0.0              0.0                 Ubaye         1.607606      1.607606
2           0.0              0.0              0.0            Parpaillon         0.435603      0.435603
600 
        abs(r1 - r2)  abs(r1 - r2)/r1  abs(r1 - r2)/r2  r1 (with mixed)  r2 (without)
count           3.0              3.0              3.0         3.000000      3.000000
mean            0.0              0.0              0.0         1.340187      1.340187
std             0.0              0.0              0.0         0.804912      0.804912
min             0.0              0.0              0.0         0.435603      0.435603
25%             0.0              0.0              0.0         1.021604      1.021604
50%             0.0              0.0              0.0         1.607606      1.607606
75%             0.0              0.0              0.0         1.792480      1.792480
max             0.0              0.0              0.0         1.977354      1.977354
   abs(r1 - r2)  abs(r1 - r2)/r1  abs(r1 - r2)/r2      massif  r1 (with mixed)  r2 (without)
0           0.0              0.0              0.0  Mercantour         2.818572      2.818572
900 
        abs(r1 - r2)  abs(r1 - r2)/r1  abs(r1 - r2)/r2  r1 (with mixed)  r2 (without)
count           1.0              1.0              1.0         1.000000      1.000000
mean            0.0              0.0              0.0         2.818572      2.818572
std             NaN              NaN              NaN              NaN           NaN
min             0.0              0.0              0.0         2.818572      2.818572
25%             0.0              0.0              0.0         2.818572      2.818572
50%             0.0              0.0              0.0         2.818572      2.818572
75%             0.0              0.0              0.0         2.818572      2.818572
max             0.0              0.0              0.0         2.818572      2.818572

Process finished with exit code 0

"""


"""
Au grand maximum: l ecart est inférieur à 0.1 kN.... entre si on utilise la mixed distribution ou non
"""

if __name__ == '__main__':
    mix_dsitrbution_impact()
