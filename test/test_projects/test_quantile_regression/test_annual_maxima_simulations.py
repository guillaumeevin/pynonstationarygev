import unittest

from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationTemporalModel
from extreme_fit.model.quantile_model.quantile_regression_model import ConstantQuantileRegressionModel, \
    TemporalCoordinatesQuantileRegressionModel
from projects.archive.quantile_regression_vs_evt.annual_maxima_simulation.daily_exp_simulation import \
    StationaryExpSimulation, NonStationaryExpSimulation
from projects.archive.quantile_regression_vs_evt.annual_maxima_simulation.gev_simulation import StationarySimulation, \
    NonStationaryLocationGumbelSimulation


class TestGevSimulations(unittest.TestCase):
    DISPLAY = False

    def test_stationary_run(self):
        simulation = StationarySimulation(nb_time_series=1, quantile=0.5, time_series_lengths=[50, 60],
                                          model_classes=[StationaryTemporalModel, ConstantQuantileRegressionModel])
        simulation.plot_error_for_last_year_quantile(self.DISPLAY)

    def test_non_stationary_run(self):
        simulation = NonStationaryLocationGumbelSimulation(nb_time_series=1, quantile=0.5, time_series_lengths=[50, 60],
                                                           model_classes=[NonStationaryLocationTemporalModel,
                                                                          TemporalCoordinatesQuantileRegressionModel])
        simulation.plot_error_for_last_year_quantile(self.DISPLAY)


class TestExpSimulations(unittest.TestCase):
    DISPLAY = False

    def test_stationary_run(self):
        simulation = StationaryExpSimulation(nb_time_series=1, quantile=0.5, time_series_lengths=[50, 60],
                                             model_classes=[StationaryTemporalModel, ConstantQuantileRegressionModel])
        simulation.plot_error_for_last_year_quantile(self.DISPLAY)

    def test_non_stationary_run(self):
        simulation = NonStationaryExpSimulation(nb_time_series=1, quantile=0.5, time_series_lengths=[50, 60],
                                                model_classes=[NonStationaryLocationTemporalModel,
                                                               TemporalCoordinatesQuantileRegressionModel])
        simulation.plot_error_for_last_year_quantile(self.DISPLAY)


if __name__ == '__main__':
    unittest.main()
