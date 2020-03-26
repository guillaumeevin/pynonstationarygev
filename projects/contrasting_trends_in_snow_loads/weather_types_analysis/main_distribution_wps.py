import pandas as pd
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad3Days, CrocusSnowLoad1Day
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranPrecipitation1Day, \
    SafranPrecipitation3Days
from extreme_data.meteo_france_data.scm_models_data.utils import SeasonForTheMaxima


def main_spatial_distribution_wps(study_class, year_min=1954, year_max=2008, limit_for_the_percentage=None):
    study = study_class(altitude=1800, year_min=year_min, year_max=year_max, season=SeasonForTheMaxima.winter_extended)
    for region_name in AbstractExtendedStudy.region_names:
        massif_names = AbstractExtendedStudy.region_name_to_massif_names[region_name]
        print('\n \n', region_name, '\n')
        for nb_top in [study.nb_years, 10, 1][1:2]:
            print(study.df_for_top_annual_maxima(nb_top=nb_top, massif_names=massif_names, limit_for_the_percentage=limit_for_the_percentage), '\n')


"""
 Northern Alps 
                 %  count  mean  std  min  median  max                                       
Steady Oceanic  92     65   159   35  121     146  282 
 
 Central Alps 
                    %  count  mean  std  min  median  max                                     
Steady Oceanic     72     51   136   31   97     130  235
South Circulation  15     11   128   18  105     126  161 

 Southern Alps 
                    %  count  mean  min  median  max                                   
South Circulation  61     37   147   79     147  235
Steady Oceanic     13      8   121   93     114  187 
 
 Extreme South Alps 
                    %  count  mean  min  median  max                                   
South Circulation  80     24   174  101     166  306 


Process finished with exit code 0






"""


def main_temporal_distribution_wps(study_class, year_min=1954, year_max=2008, limit_for_the_percentage=None):
    altitude = 1800
    study_before = study_class(altitude=altitude, year_min=year_min, year_max=1981, season=SeasonForTheMaxima.winter_extended)
    study_after = study_class(altitude=altitude, year_min=1981, year_max=year_max, season=SeasonForTheMaxima.winter_extended)
    # todo: same min and max year ?
    for region_name in AbstractExtendedStudy.region_names:
        massif_names = AbstractExtendedStudy.region_name_to_massif_names[region_name]
        print('\n \n', '{} ({} massifs)'.format(region_name, len(massif_names)), '\n')
        for nb_top in [study_before.nb_years, 10][1:]:
            print(study_before.df_for_top_annual_maxima(nb_top=nb_top, massif_names=massif_names, limit_for_the_percentage=limit_for_the_percentage), '\n')
            print(study_after.df_for_top_annual_maxima(nb_top=nb_top, massif_names=massif_names, limit_for_the_percentage=limit_for_the_percentage), '\n')

"""
 Northern Alps (7 massifs) 
                             %  count  mean  min  median  max
Top 10 maxima (1954 -1981)                                   
Steady Oceanic              94     66   130  102     127  202 

Top 10 maxima (1981 -2008)                                   
Steady Oceanic              82     58   151  104     134  282 
 
 Central Alps (7 massifs) 
                             %  count  mean  min  median  max
Top 10 maxima (1954 -1981)                                   
Steady Oceanic              71     50   110   76     107  190 

Top 10 maxima (1981 -2008)                                   
Steady Oceanic              74     52   125   87     115  235
South Circulation           14     10   123  100     120  161 

 Southern Alps (6 massifs) 
                             %  count  mean  min  median  max
Top 10 maxima (1954 -1981)                                   
South Circulation           43     26   113   67     112  197
Steady Oceanic              16     10    95   82      93  122
Southwest Circulation       15      9   102   68      95  140 

Top 10 maxima (1981 -2008)                                   
South Circulation           63     38   134   73     127  235
Steady Oceanic              21     13   110   80     105  187 


 Extreme South Alps (3 massifs) 
                             %  count  mean  min  median  max
Top 10 maxima (1954 -1981)                                   
South Circulation           63     19   136   84     139  194
Southwest Circulation       13      4   122   95     123  146 

Top 10 maxima (1981 -2008)                                   
South Circulation           76     23   165   78     159  306 




"""


if __name__ == '__main__':
    limit_percentage = 10
    study_class = [CrocusSnowLoad1Day, SafranPrecipitation1Day, SafranPrecipitation3Days][-1]
    # main_spatial_distribution_wps(study_class, limit_for_the_percentage=limit_percentage)
    main_temporal_distribution_wps(study_class, limit_for_the_percentage=limit_percentage)
