import unittest

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies


class TestAltitudesStudies(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        altitudes = [900, 1200]
        study_class = SafranSnowfall1Day
        self.studies = AltitudesStudies(study_class, altitudes, year_min=1959, year_max=1962)


class TestVisualization(TestAltitudesStudies):

    def test_plot_maxima_time_series(self):
        self.studies.plot_maxima_time_series(massif_names=['Vercors'], show=False)


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
        self.assertEqual(len(dataset.coordinates.df_coordinate_climate_model.columns), 4)


if __name__ == '__main__':
    unittest.main()
