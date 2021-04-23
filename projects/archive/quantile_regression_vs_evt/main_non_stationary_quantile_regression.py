from extreme_fit.model.margin_model.linear_margin_model.daily_data_model import TemporalCoordinatesQuantileRegressionModelOnDailyData
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationTemporalModel, NonStationaryLocationGumbelModel
from extreme_fit.model.quantile_model.quantile_regression_model import TemporalCoordinatesQuantileRegressionModel
from projects.archive.quantile_regression_vs_evt.annual_maxima_simulation.daily_exp_simulation import \
    NonStationaryExpSimulation
from projects.archive.quantile_regression_vs_evt.annual_maxima_simulation.gev_simulation import \
    NonStationaryLocationGumbelSimulation, NonStationaryLocationGevSimulation
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization, IdentityTransformation

nb_time_series = 10
quantile = 0.98
time_series_lengths = [50, 100, 200]
transformation_class = [IdentityTransformation, CenteredScaledNormalization][1]
model_classes = [
    NonStationaryLocationTemporalModel,
    TemporalCoordinatesQuantileRegressionModel,
    NonStationaryLocationGumbelModel,
    TemporalCoordinatesQuantileRegressionModelOnDailyData
]
simulation_class = [NonStationaryLocationGumbelSimulation,
                    NonStationaryLocationGevSimulation,
                    NonStationaryExpSimulation][-1]

simulation = simulation_class(nb_time_series=nb_time_series,
                              quantile=quantile,
                              time_series_lengths=time_series_lengths,
                              model_classes=model_classes,
                              transformation_class=transformation_class,
                              multiprocessing=False)
simulation.plot_error_for_last_year_quantile()
