from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel, NonStationaryShapeTemporalModel, \
    NonStationaryScaleAndShapeTemporalModel, NonStationaryLocationAndScaleAndShapeTemporalModel, \
    NonStationaryLocationAndShapeTemporalModel, NonStationaryLocationAndScaleTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationModel, NonStationaryTwoLinearShapeModel, \
    NonStationaryTwoLinearShapeOneLinearScaleModel, NonStationaryTwoLinearScaleOneLinearShapeModel, \
    NonStationaryTwoLinearScaleAndShapeModel, NonStationaryTwoLinearShapeOneLinearLocAndScaleModel, \
    NonStationaryTwoLinearScaleOneLinearLocAndShapeModel, NonStationaryTwoLinearShapeOneLinearLocModel, \
    NonStationaryTwoLinearScaleOneLinearLocModel, NonStationaryTwoLinearScaleAndShapeOneLinearLocModel, \
    NonStationaryTwoLinearLocationOneLinearScaleModel, NonStationaryTwoLinearLocationOneLinearScaleAndShapeModel, \
    NonStationaryTwoLinearLocationOneLinearShapeModel, NonStationaryTwoLinearScaleModel, \
    NonStationaryTwoLinearLocationAndShapeOneLinearScaleModel, NonStationaryTwoLinearLocationAndScaleAndShapeModel, \
    NonStationaryTwoLinearLocationAndScaleOneLinearShapeModel, NonStationaryTwoLinearLocationAndScaleModel, \
    NonStationaryTwoLinearLocationAndShape

SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE = [

    # Models with a constant Location parameter
    StationaryTemporalModel,
    # Simple linearity for the others
    NonStationaryScaleTemporalModel,
    NonStationaryShapeTemporalModel,
    NonStationaryScaleAndShapeTemporalModel,
    # Double linearity for the others
    NonStationaryTwoLinearScaleModel,
    NonStationaryTwoLinearShapeModel,
    NonStationaryTwoLinearShapeOneLinearScaleModel,
    NonStationaryTwoLinearScaleOneLinearShapeModel,
    NonStationaryTwoLinearScaleAndShapeModel,

    # Models with a linear location parameter
    NonStationaryLocationTemporalModel,
    # Simple linearity for the others
    NonStationaryLocationAndScaleTemporalModel,
    NonStationaryLocationAndShapeTemporalModel,
    NonStationaryLocationAndScaleAndShapeTemporalModel,
    # Double linearity for the others
    NonStationaryTwoLinearScaleOneLinearLocModel,
    NonStationaryTwoLinearShapeOneLinearLocModel,
    NonStationaryTwoLinearScaleOneLinearLocAndShapeModel,
    NonStationaryTwoLinearShapeOneLinearLocAndScaleModel,
    NonStationaryTwoLinearScaleAndShapeOneLinearLocModel,



    # Models with linear location parameter with double linearity
    NonStationaryTwoLinearLocationModel,
    # Simple linearity for the others
    NonStationaryTwoLinearLocationOneLinearScaleModel,
    NonStationaryTwoLinearLocationOneLinearShapeModel,
    NonStationaryTwoLinearLocationOneLinearScaleAndShapeModel,
    # Double Linearity for the others
    NonStationaryTwoLinearLocationAndScaleModel,
    NonStationaryTwoLinearLocationAndScaleOneLinearShapeModel,
    NonStationaryTwoLinearLocationAndShape,
    NonStationaryTwoLinearLocationAndShapeOneLinearScaleModel,
    NonStationaryTwoLinearLocationAndScaleAndShapeModel,


]



if __name__ == '__main__':
    print(len(set(SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE)))