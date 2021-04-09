from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationModel, NonStationaryTwoLinearScaleModel

SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE = [
    StationaryTemporalModel,

    # Location only non-stationarity
    NonStationaryLocationTemporalModel,
    NonStationaryTwoLinearLocationModel,

    # Scale only non-stationarity
    NonStationaryScaleTemporalModel,
    NonStationaryTwoLinearScaleModel,
]
