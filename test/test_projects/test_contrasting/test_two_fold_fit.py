import unittest

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import ConstantMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.utils import set_seed_for_test
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.two_fold_datasets_generator import TwoFoldDatasetsGenerator
from projects.altitude_spatial_model.altitudes_fit.two_fold_detail_fit import TwoFoldModelFit
from projects.altitude_spatial_model.altitudes_fit.two_fold_fit import TwoFoldFit
from projects.altitude_spatial_model.altitudes_fit.utils import Score
from spatio_temporal_dataset.slicer.split import Split


def load_two_fold_fit(fit_method, year_max):
    altitudes = [900, 1200]
    study_class = SafranSnowfall1Day
    studies = AltitudesStudies(study_class, altitudes, year_max=year_max)
    two_fold_datasets_generator = TwoFoldDatasetsGenerator(studies, nb_samples=1, massif_names=['Vercors'])
    model_family_name_to_model_class = {'Stationary': [ConstantMarginModel]}
    return TwoFoldFit(two_fold_datasets_generator=two_fold_datasets_generator,
                      model_family_name_to_model_classes=model_family_name_to_model_class,
                      fit_method=fit_method)


class TestTwoFoldFit(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        set_seed_for_test()

    def test_determinism_dataset_generation(self):
        two_fold_fit = load_two_fold_fit(fit_method=MarginFitMethod.spatial_extremes_mle, year_max=1963)
        massif_fit = two_fold_fit.massif_name_to_massif_fit['Vercors']
        sample_fit = massif_fit.sample_id_to_sample_fit[0]
        model_fit = sample_fit.model_class_to_model_fit[ConstantMarginModel]  # type: TwoFoldModelFit
        dataset_fold1 = model_fit.estimator_fold_1.dataset
        index_train = list(dataset_fold1.coordinates.coordinates_index(split=Split.train_temporal))
        self.assertEqual([2, 3, 8, 9], index_train)
        self.assertEqual(110.52073192596436, np.sum(dataset_fold1.maxima_gev(split=Split.train_temporal)))

    def test_determinism_fit_spatial_extreme(self):
        two_fold_fit = load_two_fold_fit(fit_method=MarginFitMethod.spatial_extremes_mle, year_max=2019)
        massif_fit = two_fold_fit.massif_name_to_massif_fit['Vercors']
        model_fit = massif_fit.sample_id_to_sample_fit[0].model_class_to_model_fit[
            ConstantMarginModel]  # type: TwoFoldModelFit
        self.assertEqual(461.6710428902022, model_fit.score(score=Score.NLLH_TEST))


if __name__ == '__main__':
    unittest.main()
