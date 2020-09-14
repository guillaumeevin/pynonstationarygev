from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import AbstractAddCrossTermForScale, \
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


class StationaryAltitudinalCrossTermScale(AbstractAddCrossTermForScale, StationaryAltitudinal):
    pass


class AltitudinalShapeConstantTimeLocationLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                               AltitudinalShapeConstantTimeLocationLinear):
    pass


class AltitudinalShapeConstantTimeScaleLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                            AltitudinalShapeConstantTimeScaleLinear):
    pass


class AltitudinalShapeConstantTimeShapeLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                            AltitudinalShapeConstantTimeShapeLinear):
    pass


class AltitudinalShapeConstantTimeLocShapeLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                               AltitudinalShapeConstantTimeLocShapeLinear):
    pass


class AltitudinalShapeConstantTimeScaleShapeLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                                 AltitudinalShapeConstantTimeScaleShapeLinear):
    pass


class AltitudinalShapeConstantTimeLocScaleLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                               AltitudinalShapeConstantTimeLocScaleLinear):
    pass


class AltitudinalShapeConstantTimeLocScaleShapeLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                                    AltitudinalShapeConstantTimeLocScaleShapeLinear):
    pass


class AltitudinalShapeLinearTimeStationaryCrossTermScale(AbstractAddCrossTermForScale,
                                                         AltitudinalShapeLinearTimeStationary):
    pass


class AltitudinalShapeLinearTimeLocationLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                             AltitudinalShapeLinearTimeLocationLinear):
    pass


class AltitudinalShapeLinearTimeScaleLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                          AltitudinalShapeLinearTimeScaleLinear):
    pass


class AltitudinalShapeLinearTimeShapeLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                          AltitudinalShapeLinearTimeShapeLinear):
    pass


class AltitudinalShapeLinearTimeLocShapeLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                             AltitudinalShapeLinearTimeLocShapeLinear):
    pass


class AltitudinalShapeLinearTimeLocScaleLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                             AltitudinalShapeLinearTimeLocScaleLinear):
    pass


class AltitudinalShapeLinearTimeScaleShapeLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                               AltitudinalShapeLinearTimeScaleShapeLinear):
    pass


class AltitudinalShapeLinearTimeLocScaleShapeLinearCrossTermScale(AbstractAddCrossTermForScale,
                                                                  AltitudinalShapeLinearTimeLocScaleShapeLinear):
    pass
