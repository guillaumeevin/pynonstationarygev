import unittest

from extreme_fit.model.daily_data_model import ConstantQuantileRegressionModelOnDailyData, \
    TemporalCoordinatesQuantileRegressionModelOnDailyData
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_exp_models import \
    NonStationaryRateTemporalModel
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationTemporalModel
from extreme_fit.model.quantile_model.quantile_regression_model import ConstantQuantileRegressionModel, \
    TemporalCoordinatesQuantileRegressionModel
from projects.quantile_regression_vs_evt.annual_maxima_simulation.daily_exp_simulation import StationaryExpSimulation, \
    NonStationaryExpSimulation
from projects.quantile_regression_vs_evt.annual_maxima_simulation.gev_simulation import StationarySimulation, \
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


class TestExpSimulationsDailyDataModels(unittest.TestCase):
    DISPLAY = False
    pass

    # Warning this method is quite long...
    # def test_stationary_run_daily_data_quantile_regression_model(self):
    #     simulation = StationaryExpSimulation(nb_time_series=1, quantile=0.5, time_series_lengths=[50, 60],
    #                                          model_classes=[ConstantQuantileRegressionModelOnDailyData])
    #     simulation.plot_error_for_last_year_quantile(self.DISPLAY)

    # def test_non_stationary_run_daily_data_quantile_regression_model(self):
    #     simulation = NonStationaryExpSimulation(nb_time_series=1, quantile=0.5, time_series_lengths=[50, 60],
    #                                             model_classes=[TemporalCoordinatesQuantileRegressionModelOnDailyData])
    #     first_estimator = simulation.model_class_to_time_series_length_to_estimators[
    #         TemporalCoordinatesQuantileRegressionModelOnDailyData][50][0]
    #     self.assertEqual(len(first_estimator.dataset.df_dataset), 50 * 365)
    #     simulation.plot_error_for_last_year_quantile(self.DISPLAY)

    # WARNING: It does not work yet, read fevd manual to understand how does he expect the parameters
    # probably the formula to provide should be w.r.t to the scale parameter
    # & there seems to be a need to be  a need to provide a threshold parameter...
    # def test_stationary_run_daily_data_exponential_model(self):
    #     simulation = StationaryExpSimulation(nb_time_series=1, quantile=0.5, time_series_lengths=[1, 2],
    #                                          model_classes=[NonStationaryRateTemporalModel])
    #     simulation.plot_error_for_last_year_quantile(self.DISPLAY)


if __name__ == '__main__':
    unittest.main()
