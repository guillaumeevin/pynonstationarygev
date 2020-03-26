import pandas as pd
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad3Days, CrocusSnowLoad1Day
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranPrecipitation1Day


def main_spatial_distribution_wps(study_class, year_min=1954, year_max=2008):
    study = study_class(altitude=1800, year_min=year_min, year_max=year_max)
    for region_name in AbstractExtendedStudy.region_names:
        massif_names = AbstractExtendedStudy.region_name_to_massif_names[region_name]
        print('\n \n', region_name, '\n')
        for nb_top in [study.nb_years, 5, 1][1:2]:
            print(study.df_for_top_annual_maxima(nb_top=nb_top, massif_names=massif_names), '\n')


"""
Some analysis:

At altitude 1800m, 
for the precipitation in 1 day
between year_min=1954, year_max=2008

If we look at the 5 biggest maxima for each massif, 
and look to which Weather pattern they correspond
then we do some percentage for each climatic region


 Northern Alps 
                      %  count  mean  std  min  median  max                                           
Steady Oceanic      77     27   105   16   80     104  150
Atlantic Wave       11      4   111   15   95     111  129

 Central Alps 
                      %  count  mean  min  median  max
Steady Oceanic      57     20    90   70      86  119
South Circulation   17      6    85   64      88  104
East Return         14      5    88   74      93  100

 Southern Alps 
                      %  count  mean  min  median  max
South Circulation   43     13    99   72     106  122
Central Depression  36     11    98   68      97  136
East Return         16      5    90   70      83  121

 Extreme South Alps 
Central Depression        53.0                     8
South Circulation         47.0                     7 

"""


def main_temporal_distribution_wps(study_class, year_min=1954, year_max=2008):
    altitude = 1800
    study_before = study_class(altitude=altitude, year_min=year_min, year_max=1981)
    study_after = study_class(altitude=altitude, year_min=1981, year_max=2008)
    for region_name in AbstractExtendedStudy.region_names:
        massif_names = AbstractExtendedStudy.region_name_to_massif_names[region_name]
        print('\n \n', region_name, '\n')
        for nb_top in [study_before.nb_years, 10][1:]:
            print(study_before.df_for_top_annual_maxima(nb_top=nb_top, massif_names=massif_names), '\n')
            print(study_after.df_for_top_annual_maxima(nb_top=nb_top, massif_names=massif_names), '\n')

"""
There is no real stationarity in the percentage of the kind of storms that are causing extreme.

the reparittion of storm before and after (no matter the nb_top consider 10, or all) are the same.
even for the local region it is the same.

-> so what is really changing is probably the strength associated to each kind of storm.
"""


if __name__ == '__main__':
    study_class = [CrocusSnowLoad1Day, SafranPrecipitation1Day][-1]
    main_spatial_distribution_wps(study_class)
    # main_temporal_distribution_wps(study_class)
