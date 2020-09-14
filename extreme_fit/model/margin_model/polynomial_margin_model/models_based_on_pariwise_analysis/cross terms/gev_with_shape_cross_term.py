from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import AbstractAddCrossTermForShape, \
    StationaryAltitudinal
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_constant_shape_wrt_altitude import \
    AltitudinalShapeConstantTimeLocationLinear, AltitudinalShapeConstantTimeScaleLinear, \
    AltitudinalShapeConstantTimeShapeLinear, AltitudinalShapeConstantTimeLocShapeLinear, \
    AltitudinalShapeConstantTimeScaleShapeLinear, AltitudinalShapeConstantTimeLocScaleLinear, \
    AltitudinalShapeConstantTimeLocScaleShapeLinear
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_linear_shape_wrt_altitude import \
    AltitudinalShapeLinearTimeLocScaleLinear, AltitudinalShapeLinearTimeLocationLinear, \
    AltitudinalShapeLinearTimeScaleLinear, AltitudinalShapeLinearTimeShapeLinear, \
    AltitudinalShapeLinearTimeLocShapeLinear, AltitudinalShapeLinearTimeScaleShapeLinear, \
    AltitudinalShapeLinearTimeLocScaleShapeLinear, AltitudinalShapeLinearTimeStationary


class StationaryAltitudinalCrossTermShape(AbstractAddCrossTermForShape, StationaryAltitudinal):
    pass


class AltitudinalShapeConstantTimeLocationLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                               AltitudinalShapeConstantTimeLocationLinear):
    pass


class AltitudinalShapeConstantTimeScaleLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                            AltitudinalShapeConstantTimeScaleLinear):
    pass


class AltitudinalShapeConstantTimeShapeLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                            AltitudinalShapeConstantTimeShapeLinear):
    pass


class AltitudinalShapeConstantTimeLocShapeLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                               AltitudinalShapeConstantTimeLocShapeLinear):
    pass


class AltitudinalShapeConstantTimeScaleShapeLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                                 AltitudinalShapeConstantTimeScaleShapeLinear):
    pass


class AltitudinalShapeConstantTimeLocScaleLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                               AltitudinalShapeConstantTimeLocScaleLinear):
    pass


class AltitudinalShapeConstantTimeLocScaleShapeLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                                    AltitudinalShapeConstantTimeLocScaleShapeLinear):
    pass


class AltitudinalShapeLinearTimeStationaryCrossTermShape(AbstractAddCrossTermForShape,
                                                         AltitudinalShapeLinearTimeStationary):
    pass


class AltitudinalShapeLinearTimeLocationLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                             AltitudinalShapeLinearTimeLocationLinear):
    pass


class AltitudinalShapeLinearTimeScaleLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                          AltitudinalShapeLinearTimeScaleLinear):
    pass


class AltitudinalShapeLinearTimeShapeLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                          AltitudinalShapeLinearTimeShapeLinear):
    pass


class AltitudinalShapeLinearTimeLocShapeLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                             AltitudinalShapeLinearTimeLocShapeLinear):
    pass


class AltitudinalShapeLinearTimeLocScaleLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                             AltitudinalShapeLinearTimeLocScaleLinear):
    pass


class AltitudinalShapeLinearTimeScaleShapeLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                               AltitudinalShapeLinearTimeScaleShapeLinear):
    pass


class AltitudinalShapeLinearTimeLocScaleShapeLinearCrossTermShape(AbstractAddCrossTermForShape,
                                                                  AltitudinalShapeLinearTimeLocScaleShapeLinear):
    pass
