import unittest

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from spatio_temporal_dataset.slicer.split import Split


class TestAltitudesStudies(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        altitudes = [900, 1200]
        study_class = SafranSnowfall1Day
        self.studies = AltitudesStudies(study_class, altitudes, year_min=1959, year_max=1962)


class TestVisualization(TestAltitudesStudies):

    def test_plot_maxima_time_series(self):
        self.studies.plot_maxima_time_series(massif_names=['Vercors'], show=False)


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


class TestSpatioTemporalDatasetForClimateModels(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        altitudes = [900, 1200]
        study_class = AdamontSnowfall
        self.scenario = AdamontScenario.rcp85
        self.studies = AltitudesStudies(study_class, altitudes,
                                        year_min=2009, year_max=2012,
                                        scenario=self.scenario)
        self.massif_name = "Vercors"

    def test_dataset(self):
        dataset = self.studies.spatio_temporal_dataset(self.massif_name)
        self.assertEqual(self.studies.study.scenario, AdamontScenario.rcp85)
        self.assertEqual(len(dataset.coordinates.df_coordinate_climate_model.columns), 3)

if __name__ == '__main__':
    unittest.main()
