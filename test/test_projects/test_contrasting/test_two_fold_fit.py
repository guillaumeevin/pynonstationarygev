import unittest
import unittest

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import ConstantMarginModel, \
    LinearLocationAllDimsMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.utils import set_seed_for_test
from projects.contrasting_trends_in_snow_loads.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.contrasting_trends_in_snow_loads.altitudes_fit.two_fold_datasets_generator import TwoFoldDatasetsGenerator
from projects.contrasting_trends_in_snow_loads.altitudes_fit.two_fold_fit import TwoFoldFit
from spatio_temporal_dataset.slicer.split import Split


class TestTwoFoldFit(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        set_seed_for_test()
        altitudes = [900, 1200]
        study_class = SafranSnowfall1Day
        studies = AltitudesStudies(study_class, altitudes, year_min=1959, year_max=1989)
        self.two_fold_datasets_generator = TwoFoldDatasetsGenerator(studies, nb_samples=2, massif_names=['Vercors'])
        self.model_family_name_to_model_class = {'Stationary': [ConstantMarginModel],
                                                 'Linear': [ConstantMarginModel, LinearLocationAllDimsMarginModel]}

    def load_two_fold_fit(self, fit_method):
        return TwoFoldFit(two_fold_datasets_generator=self.two_fold_datasets_generator,
                          model_family_name_to_model_classes=self.model_family_name_to_model_class,
                          fit_method=fit_method)

    def test_best_fit_spatial_extreme(self):
        two_fold_fit = self.load_two_fold_fit(fit_method=MarginFitMethod.spatial_extremes_mle)
        try:
            best_model_class = two_fold_fit.massif_name_to_best_model()['Vercors']
        except AssertionError as e:
            self.assertTrue(False, msg=e.__str__())
        self.assertEqual(best_model_class, LinearLocationAllDimsMarginModel)


if __name__ == '__main__':
    unittest.main()
