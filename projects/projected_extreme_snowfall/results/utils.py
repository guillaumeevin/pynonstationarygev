from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel, NonStationaryShapeTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationModel, NonStationaryTwoLinearShapeModel

SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE = [
    # 1 model with three parameters
    StationaryTemporalModel,

    # 3 models with four parameters
    NonStationaryLocationTemporalModel,
    NonStationaryScaleTemporalModel,
    NonStationaryShapeTemporalModel,


    # Location only non-stationarity

    NonStationaryTwoLinearLocationModel,

    # Scale only non-stationarity

    NonStationaryTwoLinearShapeModel,

    # Shape only non-stationarity


    #
]
