from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationTemporalModel, NonStationaryLocationGumbelModel
from extreme_fit.model.quantile_model.quantile_regression_model import ConstantQuantileRegressionModel, \
    TemporalCoordinatesQuantileRegressionModel
from projects.quantile_regression_vs_evt.GevSimulation import StationarySimulation, \
    NonStationaryLocationGumbelSimulation, NonStationaryLocationGevSimulation
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization, IdentityTransformation
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization

nb_time_series = 10
quantile = 0.98
time_series_lengths = [50, 100, 200]
transformation_class = [IdentityTransformation, CenteredScaledNormalization][0]
model_classes = [NonStationaryLocationTemporalModel, TemporalCoordinatesQuantileRegressionModel, NonStationaryLocationGumbelModel]
simulation_class = [NonStationaryLocationGumbelSimulation, NonStationaryLocationGevSimulation][0]

simulation = NonStationaryLocationGumbelSimulation(nb_time_series=nb_time_series,
                                                   quantile=quantile,
                                                   time_series_lengths=time_series_lengths,
                                                   model_classes=model_classes,
                                                   transformation_class=transformation_class)
simulation.plot_error_for_last_year_quantile()
