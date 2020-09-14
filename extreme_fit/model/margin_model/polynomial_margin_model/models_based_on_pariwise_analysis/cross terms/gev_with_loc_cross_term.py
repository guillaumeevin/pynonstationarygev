from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import AbstractAddCrossTermForLocation, \
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


class StationaryAltitudinalCrossTermLoc(AbstractAddCrossTermForLocation, StationaryAltitudinal):
    pass


class AltitudinalShapeConstantTimeLocationLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                               AltitudinalShapeConstantTimeLocationLinear):
    pass


class AltitudinalShapeConstantTimeScaleLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                            AltitudinalShapeConstantTimeScaleLinear):
    pass


class AltitudinalShapeConstantTimeShapeLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                            AltitudinalShapeConstantTimeShapeLinear):
    pass


class AltitudinalShapeConstantTimeLocShapeLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                               AltitudinalShapeConstantTimeLocShapeLinear):
    pass


class AltitudinalShapeConstantTimeScaleShapeLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                                 AltitudinalShapeConstantTimeScaleShapeLinear):
    pass


class AltitudinalShapeConstantTimeLocScaleLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                               AltitudinalShapeConstantTimeLocScaleLinear):
    pass


class AltitudinalShapeConstantTimeLocScaleShapeLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                                    AltitudinalShapeConstantTimeLocScaleShapeLinear):
    pass


class AltitudinalShapeLinearTimeStationaryCrossTermLoc(AbstractAddCrossTermForLocation,
                                                         AltitudinalShapeLinearTimeStationary):
    pass


class AltitudinalShapeLinearTimeLocationLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                             AltitudinalShapeLinearTimeLocationLinear):
    pass


class AltitudinalShapeLinearTimeScaleLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                          AltitudinalShapeLinearTimeScaleLinear):
    pass


class AltitudinalShapeLinearTimeShapeLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                          AltitudinalShapeLinearTimeShapeLinear):
    pass


class AltitudinalShapeLinearTimeLocShapeLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                             AltitudinalShapeLinearTimeLocShapeLinear):
    pass


class AltitudinalShapeLinearTimeLocScaleLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                             AltitudinalShapeLinearTimeLocScaleLinear):
    pass


class AltitudinalShapeLinearTimeScaleShapeLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                               AltitudinalShapeLinearTimeScaleShapeLinear):
    pass


class AltitudinalShapeLinearTimeLocScaleShapeLinearCrossTermLoc(AbstractAddCrossTermForLocation,
                                                                  AltitudinalShapeLinearTimeLocScaleShapeLinear):
    pass
