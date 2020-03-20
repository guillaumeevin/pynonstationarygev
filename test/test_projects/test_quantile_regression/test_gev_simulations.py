import unittest

from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.quantile_model.quantile_regression_model import ConstantQuantileRegressionModel
from projects.quantile_regression_vs_evt.GevSimulation import GevSimulation, StationarySimulation


class TestGevSimulations(unittest.TestCase):
    DISPLAY = False

    def test_stationary_run(self):
        simulation = StationarySimulation(nb_time_series=1, quantile=0.5, time_series_lengths=[50, 60],
                                          model_classes=[StationaryTemporalModel, ConstantQuantileRegressionModel])
        simulation.plot_error_for_last_year_quantile(self.DISPLAY)


if __name__ == '__main__':
    unittest.main()
