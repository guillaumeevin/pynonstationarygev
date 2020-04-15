import unittest
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from projects.contrasting_trends_in_snow_loads.altitunal_fit.altitudes_studies import AltitudesStudies
from projects.contrasting_trends_in_snow_loads.altitunal_fit.two_fold_estimation import TwoFoldEstimation
from spatio_temporal_dataset.slicer.split import Split


class TestAltitudesStudies(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        altitudes = [900, 1200]
        study_class = SafranSnowfall1Day
        studies = AltitudesStudies(study_class, altitudes, year_min=1959, year_max=1962)
        self.two_fold_estimation = TwoFoldEstimation(studies, nb_samples=2)

    def test_dataset_sizes(self):
        dataset1, dataset2 = self.two_fold_estimation.two_fold_datasets('Vercors')
        np.testing.assert_equal(dataset1.maxima_gev(Split.train_temporal), dataset2.maxima_gev(Split.test_temporal))
        np.testing.assert_equal(dataset1.maxima_gev(Split.test_temporal), dataset2.maxima_gev(Split.train_temporal))

    def test_crash(self):
        dataset1, _ = self.two_fold_estimation.two_fold_datasets('Vercors')
        with self.assertRaises(AssertionError):
            dataset1.maxima_gev(split=Split.train_spatiotemporal)
        with self.assertRaises(AssertionError):
            dataset1.maxima_gev(split=Split.train_spatial)


if __name__ == '__main__':
    unittest.main()
