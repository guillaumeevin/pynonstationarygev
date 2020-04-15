import unittest

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from projects.contrasting_trends_in_snow_loads.altitunal_fit.altitudes_studies import AltitudesStudies
from spatio_temporal_dataset.slicer.split import Split, small_s_split_from_ratio
from spatio_temporal_dataset.slicer.temporal_slicer import TemporalSlicer
import pandas as pd


class TestAltitudesStudies(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        altitudes = [900, 1200]
        study_class = SafranSnowfall1Day
        self.studies = AltitudesStudies(study_class, altitudes, year_min=1959, year_max=1962)

    def test_spatio_temporal_coordinates_with_temporal_split(self):
        s_split_temporal = self.studies.random_s_split_temporal(train_split_ratio=0.75)
        coordinates = self.studies.spatio_temporal_coordinates(slicer_class=TemporalSlicer,
                                                               s_split_temporal=s_split_temporal)
        train_values = coordinates.coordinates_values(split=Split.train_temporal)
        self.assertEqual(train_values.shape, (6, 2))
        test_values = coordinates.coordinates_values(split=Split.test_temporal)
        self.assertEqual(test_values.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()