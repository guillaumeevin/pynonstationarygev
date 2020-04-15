import unittest

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from projects.contrasting_trends_in_snow_loads.altitunal_fit.altitudes_studies import AltitudesStudies
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.split import Split, small_s_split_from_ratio
from spatio_temporal_dataset.slicer.temporal_slicer import TemporalSlicer
import pandas as pd


class TestAltitudesStudies(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        altitudes = [900, 1200]
        study_class = SafranSnowfall1Day
        self.studies = AltitudesStudies(study_class, altitudes, year_min=1959, year_max=1962)


class TestSpatioTemporalCoordinates(TestAltitudesStudies):

    def test_temporal_split(self):
        s_split_temporal = self.studies.random_s_split_temporal(train_split_ratio=0.75)
        coordinates = self.studies.spatio_temporal_coordinates(s_split_temporal=s_split_temporal)
        self.assertEqual(coordinates.coordinates_values(split=Split.train_temporal).shape, (6, 2))
        self.assertEqual(coordinates.coordinates_values(split=Split.test_temporal).shape, (2, 2))

    def test_spatial_split(self):
        s_split_spatial = self.studies.random_s_split_spatial(train_split_ratio=0.5)
        coordinates = self.studies.spatio_temporal_coordinates(s_split_spatial=s_split_spatial)
        self.assertEqual(coordinates.coordinates_values(split=Split.train_spatial).shape, (4, 2))
        self.assertEqual(coordinates.coordinates_values(split=Split.test_spatial).shape, (4, 2))

    def test_spatio_temporal_split(self):
        s_split_spatial = self.studies.random_s_split_spatial(train_split_ratio=0.5)
        s_split_temporal = self.studies.random_s_split_temporal(train_split_ratio=0.75)
        coordinates = self.studies.spatio_temporal_coordinates(s_split_spatial=s_split_spatial,
                                                               s_split_temporal=s_split_temporal)
        self.assertEqual(coordinates.coordinates_values(split=Split.train_spatiotemporal).shape, (3, 2))
        self.assertEqual(coordinates.coordinates_values(split=Split.test_spatiotemporal_spatial).shape, (3, 2))
        self.assertEqual(coordinates.coordinates_values(split=Split.test_spatiotemporal_temporal).shape, (1, 2))
        self.assertEqual(coordinates.coordinates_values(split=Split.test_spatiotemporal).shape, (1, 2))


class TestSpatioTemporalDataset(TestAltitudesStudies):

    def setUp(self) -> None:
        super().setUp()
        self.massif_name = "Vercors"

    def test_temporal_split(self):
        s_split_temporal = self.studies.random_s_split_temporal(train_split_ratio=0.75)
        dataset = self.studies.spatio_temporal_dataset(massif_name=self.massif_name,
                                                       s_split_temporal=s_split_temporal)
        self.assertEqual(len(dataset.maxima_gev(split=Split.train_temporal)), 6)
        self.assertEqual(len(dataset.maxima_gev(split=Split.test_temporal)), 2)

    def test_spatial_split(self):
        s_split_spatial = self.studies.random_s_split_spatial(train_split_ratio=0.5)
        dataset = self.studies.spatio_temporal_dataset(massif_name=self.massif_name,
                                                       s_split_spatial=s_split_spatial)
        self.assertEqual(len(dataset.maxima_gev(split=Split.train_spatial)), 4)
        self.assertEqual(len(dataset.maxima_gev(split=Split.test_spatial)), 4)

    def test_spatio_temporal_split(self):
        s_split_spatial = self.studies.random_s_split_spatial(train_split_ratio=0.5)
        s_split_temporal = self.studies.random_s_split_temporal(train_split_ratio=0.75)
        dataset = self.studies.spatio_temporal_dataset(massif_name=self.massif_name,
                                                       s_split_spatial=s_split_spatial,
                                                       s_split_temporal=s_split_temporal)
        self.assertEqual(len(dataset.maxima_gev(split=Split.train_spatiotemporal)), 3)
        self.assertEqual(len(dataset.maxima_gev(split=Split.test_spatiotemporal)), 1)
        self.assertEqual(len(dataset.maxima_gev(split=Split.test_spatiotemporal_temporal)), 1)
        self.assertEqual(len(dataset.maxima_gev(split=Split.test_spatiotemporal_spatial)), 3)


if __name__ == '__main__':
    unittest.main()
