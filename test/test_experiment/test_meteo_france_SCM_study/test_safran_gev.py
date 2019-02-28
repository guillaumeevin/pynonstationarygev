import unittest
import os
import os.path as op
from collections import OrderedDict
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

from experiment.meteo_france_SCM_study.abstract_variable import AbstractVariable
from experiment.meteo_france_SCM_study.massif import safran_massif_names_from_datasets
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima
from utils import get_full_path, cached_property
#
# from test.test_utils import load_safran_objects
#
#
# class TestFullEstimators(unittest.TestCase):
#
#     def test_gev_mle_per_massif(self):
#         safran_1800_one_day = load_safran_objects()[0]
#         df = safran_1800_one_day.df_gev_mle_each_massif
#         self.assertAlmostEqual(df.values.sum(), 1131.4551665871832)
#
#
# if __name__ == '__main__':
#     unittest.main()
