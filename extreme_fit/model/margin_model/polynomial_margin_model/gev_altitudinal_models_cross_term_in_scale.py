from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import AbstractAltitudinalModel, \
    NonStationaryAltitudinalScaleLinear, NonStationaryAltitudinalLocationLinearScaleLinear, \
    NonStationaryAltitudinalLocationQuadraticScaleLinear, AbstractAddCrossTermForScale
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models_with_scale import \
    NonStationaryAltitudinalLocationQuadraticScaleQuadratic, NonStationaryAltitudinalScaleQuadratic, \
    NonStationaryAltitudinalLocationLinearScaleQuadratic
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import PolynomialMarginModel
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel




class NonStationaryAltitudinalScaleLinearCrossTermForScale(AbstractAddCrossTermForScale,
                                                              NonStationaryAltitudinalScaleLinear):
    pass


class NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForScale(AbstractAddCrossTermForScale,
                                                                            NonStationaryAltitudinalLocationLinearScaleLinear):
    pass


class NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForScale(AbstractAddCrossTermForScale,
                                                                               NonStationaryAltitudinalLocationQuadraticScaleLinear):
    pass

class NonStationaryAltitudinalScaleQuadraticCrossTermForScale(AbstractAddCrossTermForScale, NonStationaryAltitudinalScaleQuadratic):
    pass


class NonStationaryAltitudinalLocationLinearScaleQuadraticCrossTermForScale(AbstractAddCrossTermForScale,
                                                                 NonStationaryAltitudinalLocationLinearScaleQuadratic):
    pass


class NonStationaryAltitudinalLocationQuadraticScaleQuadraticCrossTermForScale(AbstractAddCrossTermForScale,
                                                                    NonStationaryAltitudinalLocationQuadraticScaleQuadratic):
    pass





