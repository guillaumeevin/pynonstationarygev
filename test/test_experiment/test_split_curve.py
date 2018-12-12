# import unittest
#
#
# from experiment.simulation.abstract_simulation import AbstractSimulation
# from extreme_estimator.extreme_models.margin_model.smooth_margin_model import ConstantMarginModel, \
#     LinearAllParametersAllDimsMarginModel
# from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
# from extreme_estimator.gev_params import GevParams
# from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import LinSpaceSpatialCoordinates
# from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset
#
#
# class TestSplitCurve(unittest.TestCase):
#     DISPLAY = False
#
#     class SplitCurveFastForTest(AbstractSimulation):
#
#         def __init__(self, nb_fit: int = 1):
#             super().__init__(nb_fit)
#             self.nb_points = 50
#             self.nb_obs = 60
#             self.coordinates = LinSpaceSpatialCoordinates.from_nb_points(nb_points=self.nb_points, train_split_ratio=0.8)
#             # MarginModel Linear with respect to the shape (from 0.01 to 0.02)
#             params_sample = {
#                 (GevParams.GEV_LOC, 0): 10,
#                 (GevParams.GEV_SHAPE, 0): 1.0,
#                 (GevParams.GEV_SCALE, 0): 1.0,
#             }
#             self.margin_model = ConstantMarginModel(coordinates=self.coordinates, params_sample=params_sample)
#             self.max_stable_model = Smith()
#
#         def load_dataset(self):
#             return FullSimulatedDataset.from_double_sampling(nb_obs=self.nb_obs, margin_model=self.margin_model,
#                                                              coordinates=self.coordinates,
#                                                              max_stable_model=self.max_stable_model)
#
#     def test_split_curve(self):
#         s = self.SplitCurveFastForTest(nb_fit=2)
#         s.fit(show=self.DISPLAY)
#
#
# if __name__ == '__main__':
#     unittest.main()
