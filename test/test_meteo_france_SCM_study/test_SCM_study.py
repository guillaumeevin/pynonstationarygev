import os
import os.path as op
from collections import OrderedDict
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

from experiment.meteo_france_SCM_study.abstract_variable import AbstractVariable
from experiment.meteo_france_SCM_study.massif import safran_massif_names_from_datasets
from experiment.meteo_france_SCM_study.safran.safran import Safran
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima
from utils import get_full_path, cached_property
import unittest

from test.test_utils import load_safran_objects


class TestSCMStudy(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.study = Safran()

    def test_massif_safran(self):
        df_centroid = pd.read_csv(op.join(self.study.map_full_path, 'coordonnees_massifs_alpes.csv'))
        # Assert that the massif names are the same between SAFRAN and the coordinate file
        assert not set(self.study.safran_massif_names).symmetric_difference(set(df_centroid['NOM']))


if __name__ == '__main__':
    unittest.main()
