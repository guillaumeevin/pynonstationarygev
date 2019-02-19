from typing import List

import os.path as op
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd

from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.abstract_variable import AbstractVariable
from experiment.meteo_france_SCM_study.massif import safran_massif_names_from_datasets
from experiment.meteo_france_SCM_study.safran.safran_snowfall_variable import SafranSnowfallVariable
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima
from utils import cached_property


class Safran(AbstractStudy):

    def __init__(self, safran_altitude=1800, nb_days_of_snowfall=1):
        super().__init__(safran_altitude)
        self.nb_days_of_snowfall = nb_days_of_snowfall
        self.model_name = 'Safran'
        self.variable_class = SafranSnowfallVariable


    def instantiate_variable_object(self, dataset) -> AbstractVariable:
        return self.variable_class(dataset, self.nb_days_of_snowfall)


