from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import GumbelTemporalModel
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import AbstractAltitudinalModel, \
    StationaryAltitudinal, NonStationaryAltitudinalScaleLinear, NonStationaryAltitudinalLocationLinear, \
    NonStationaryAltitudinalLocationQuadratic, NonStationaryAltitudinalLocationLinearScaleLinear, \
    NonStationaryAltitudinalLocationQuadraticScaleLinear, NonStationaryAltitudinalScaleLinearCrossTermForLocation, \
    NonStationaryCrossTermForLocation, NonStationaryAltitudinalLocationLinearCrossTermForLocation, \
    NonStationaryAltitudinalLocationQuadraticCrossTermForLocation, \
    NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForLocation, \
    NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation, AbstractAddCrossTermForLocation
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import PolynomialMarginModel
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractGumbelAltitudinalModel(AbstractAltitudinalModel):
    DISTRIBUTION_STR = 'Gum'

    def __init__(self, *args, max_degree=4, **kwargs):
        super().__init__(*args, max_degree=4, **kwargs, type_for_MLE="Gumbel")

class StationaryGumbelAltitudinal(AbstractGumbelAltitudinalModel, StationaryAltitudinal):
    pass


class NonStationaryGumbelAltitudinalScaleLinear(AbstractGumbelAltitudinalModel, NonStationaryAltitudinalScaleLinear):
    pass


class NonStationaryGumbelAltitudinalLocationLinear(AbstractGumbelAltitudinalModel,
                                                   NonStationaryAltitudinalLocationLinear):
    pass


class NonStationaryGumbelAltitudinalLocationQuadratic(AbstractGumbelAltitudinalModel,
                                                      NonStationaryAltitudinalLocationQuadratic):
    pass


class NonStationaryGumbelAltitudinalLocationLinearScaleLinear(AbstractGumbelAltitudinalModel,
                                                              NonStationaryAltitudinalLocationLinearScaleLinear):
    pass


class NonStationaryGumbelAltitudinalLocationQuadraticScaleLinear(AbstractGumbelAltitudinalModel,
                                                                 NonStationaryAltitudinalLocationQuadraticScaleLinear):
    pass


# Add cross terms


class NonStationaryGumbelCrossTermForLocation(AbstractGumbelAltitudinalModel,
                                              NonStationaryCrossTermForLocation):
    pass


class NonStationaryGumbelAltitudinalLocationLinearCrossTermForLocation(AbstractGumbelAltitudinalModel,
                                                                       NonStationaryAltitudinalLocationLinearCrossTermForLocation):
    pass


class NonStationaryGumbelAltitudinalLocationQuadraticCrossTermForLocation(AbstractGumbelAltitudinalModel,
                                                                          NonStationaryAltitudinalLocationQuadraticCrossTermForLocation):
    pass


class NonStationaryGumbelAltitudinalLocationLinearScaleLinearCrossTermForLocation(AbstractGumbelAltitudinalModel,
                                                                                  NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForLocation):
    pass


class NonStationaryGumbelAltitudinalLocationQuadraticScaleLinearCrossTermForLocation(AbstractGumbelAltitudinalModel,
                                                                                     NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation):
    pass


class NonStationaryGumbelAltitudinalScaleLinearCrossTermForLocation(AbstractGumbelAltitudinalModel,
                                                                    NonStationaryAltitudinalScaleLinearCrossTermForLocation):
    pass
