import pandas as pd
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad3Days, CrocusSnowLoad1Day
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranPrecipitation1Day


def main_spatial_distribution_wps(study_class, year_min=1954, year_max=2008):
    study = study_class(altitude=1800, year_min=year_min, year_max=year_max)
    for region_name in AbstractExtendedStudy.region_names:
        massifs_ids = AbstractExtendedStudy.region_name_to_massif_ids[region_name]
        print('\n \n', region_name, '\n')
        for nb_top in [study.nb_years, 5, 1][1:2]:
            print(study.wps_for_top_annual_maxima(nb_top=nb_top, massif_ids=massifs_ids), '\n')


"""
Some analysis:

At altitude 1800m, 
for the precipitation in 1 day
between year_min=1954, year_max=2008

If we look at the 5 biggest maxima for each massif, 
and look to which Weather pattern they correspond
then we do some percentage for each climatic region

                    Percentage  Nb massifs concerned
 Northern Alps 
Steady Oceanic            77.0                    27
Atlantic Wave             11.0                     4

 Central Alps 
Steady Oceanic            57.0                    20
South Circulation         17.0                     6
East Return               14.0                     5

 Southern Alps 
South Circulation         43.0                    13
Central Depression        37.0                    11
East Return               17.0                     5

 Extreme South Alps 
Central Depression        53.0                     8
South Circulation         47.0                     7 

"""


def main_temporal_distribution_wps(study_class, year_min=1954, year_max=2008):
    altitude = 1800
    study_before = study_class(altitude=altitude, year_min=year_min, year_max=1981)
    study_after = study_class(altitude=altitude, year_min=1981, year_max=2008)
    for region_name in AbstractExtendedStudy.region_names:
        massifs_ids = AbstractExtendedStudy.region_name_to_massif_ids[region_name]
        print('\n \n', region_name, '\n')
        for nb_top in [study_before.nb_years, 10, 5, 1][1:2]:
            print(study_before.wps_for_top_annual_maxima(nb_top=nb_top, massif_ids=massifs_ids), '\n')
            print(study_after.wps_for_top_annual_maxima(nb_top=nb_top, massif_ids=massifs_ids), '\n')

"""
There is no real stationarity in the percentage of the kind of storms that are causing extreme.

the reparittion of storm before and after (no matter the nb_top consider 10, or all) are the same.
even for the local region it is the same.

-> so what is really changing is probably the strength associated to each kind of storm.
"""


if __name__ == '__main__':
    study_class = [CrocusSnowLoad1Day, SafranPrecipitation1Day][-1]
    # main_spatial_distribution_wps(study_class)
    main_temporal_distribution_wps(study_class)
