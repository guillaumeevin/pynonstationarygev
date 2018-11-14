import unittest

from extreme_estimator.R_model.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.R_model.margin_model.smooth_margin_model import ConstantMarginModel
from extreme_estimator.estimator.margin_estimator import SmoothMarginEstimator
from spatio_temporal_dataset.dataset.simulation_dataset import MarginDataset
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1


class TestMarginEstimators(unittest.TestCase):
    DISPLAY = False
    MARGIN_TYPES = [ConstantMarginModel]
    MARGIN_ESTIMATORS = [SmoothMarginEstimator]

    def setUp(self):
        super().setUp()
        self.spatial_coordinates = CircleCoordinatesRadius1.from_nb_points(nb_points=5, max_radius=1)
        self.margin_models = self.load_margin_models(spatial_coordinates=self.spatial_coordinates)

    @classmethod
    def load_margin_models(cls, spatial_coordinates):
        return [margin_class(spatial_coordinates=spatial_coordinates) for margin_class in cls.MARGIN_TYPES]

    def test_dependency_estimators(self):
        for margin_model in self.margin_models:
            dataset = MarginDataset.from_sampling(nb_obs=10, margin_model=margin_model,
                                                  spatial_coordinates=self.spatial_coordinates)

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
