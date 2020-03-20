from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationTemporalModel
from extreme_fit.model.quantile_model.quantile_regression_model import ConstantQuantileRegressionModel, \
    TemporalCoordinatesQuantileRegressionModel
from projects.quantile_regression_vs_evt.GevSimulation import StationarySimulation, NonStationaryLocationSimulation

nb_time_series = 10
quantile = 0.9
time_series_lengths = [50, 100, 200]

# simulation = StationarySimulation(nb_time_series=nb_time_series, quantile=quantile, time_series_lengths=time_series_lengths,
#                                   model_classes=[StationaryTemporalModel, ConstantQuantileRegressionModel])
# simulation.plot_error_for_last_year_quantile()

simulation = NonStationaryLocationSimulation(nb_time_series=nb_time_series, quantile=quantile, time_series_lengths=time_series_lengths,
                                  model_classes=[NonStationaryLocationTemporalModel, TemporalCoordinatesQuantileRegressionModel][:])
simulation.plot_error_for_last_year_quantile()
