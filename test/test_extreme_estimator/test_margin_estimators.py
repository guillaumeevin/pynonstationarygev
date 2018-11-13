import unittest

import numpy as np

from extreme_estimator.R_model.margin_model.abstract_margin_model import ConstantMarginModel
from extreme_estimator.R_model.margin_model.gev_mle_fit import GevMleFit
from extreme_estimator.R_model.utils import get_loaded_r
from extreme_estimator.estimator.margin_estimator import SmoothMarginEstimator
from spatio_temporal_dataset.dataset.simulation_dataset import MarginDataset
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1


class TestMarginEstimators(unittest.TestCase):
    DISPLAY = False
    MARGIN_TYPES = [ConstantMarginModel]
    MARGIN_ESTIMATORS = [SmoothMarginEstimator]

    def test_unitary_mle_gev_fit(self):
        r = get_loaded_r()
        r("""
        set.seed(42)
        N <- 50
        loc = 0; scale = 1; shape <- 1
        x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
        start_loc = 0; start_scale = 1; start_shape = 1
        """)
        # Get the MLE estimator
        estimator = GevMleFit(x_gev=np.array(r['x_gev']),
                              start_loc=np.float(r['start_loc'][0]),
                              start_scale=np.float(r['start_scale'][0]),
                              start_shape=np.float(r['start_shape'][0]))
        # Compare the MLE estimated parameters to the reference
        mle_params_estimated = estimator.mle_params
        mle_params_ref = {'loc': 0.0219, 'scale': 1.0347, 'shape': 0.8290}
        for key in mle_params_ref.keys():
            self.assertAlmostEqual(mle_params_ref[key], mle_params_estimated[key], places=3)

    def setUp(self):
        super().setUp()
        self.spatial_coord = CircleCoordinatesRadius1.from_nb_points(nb_points=5, max_radius=1)
        self.margin_models = self.load_margin_models()

    @classmethod
    def load_margin_models(cls):
        return [margin_class() for margin_class in cls.MARGIN_TYPES]

    def test_dependency_estimators(self):
        for margin_model in self.margin_models:
            dataset = MarginDataset.from_sampling(nb_obs=10, margin_model=margin_model,
                                                  spatial_coordinates=self.spatial_coord)

            for estimator_class in self.MARGIN_ESTIMATORS:
                estimator = estimator_class(dataset=dataset, margin_model=margin_model)
                estimator.fit()
                if self.DISPLAY:
                    print(type(margin_model))
                    print(dataset.df_dataset.head())
                    print(estimator.additional_information)
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
