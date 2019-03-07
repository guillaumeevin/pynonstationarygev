import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiment.meteo_france_SCM_study.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_SCM_study.visualization.studies_visualization.studies import \
    Studies
from experiment.meteo_france_SCM_study.visualization.utils import plot_df


class StudiesVisualizer(object):

    def __init__(self, studies: Studies) -> None:
        self.studies = studies

    @property
    def first_study(self):
        return self.studies.first_study

    def mean_as_a_function_of_altitude(self, region_only=False):
        # Load the massif names to display
        if region_only:
            assert isinstance(self.first_study, AbstractExtendedStudy)
            massif_names = self.first_study.region_names
        else:
            massif_names = self.first_study.safran_massif_names
        # Load the dictionary that maps each massif_name to its corresponding time series
        mean_series = []
        for study in self.studies.altitude_to_study.values():
            mean_serie = study.df_annual_total.loc[:, massif_names].mean(axis=0)
            mean_series.append(mean_serie)
        df_mean = pd.concat(mean_series, axis=1) # type: pd.DataFrame
        df_mean.columns = self.studies.altitude_list
        plot_df(df_mean)





