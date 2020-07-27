from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import AbstractAltitudinalModel, \
    AbstractAddCrossTermForLocation
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import PolynomialMarginModel
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel






class NonStationaryAltitudinalScaleQuadratic(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
        }

class NonStationaryAltitudinalLocationLinearScaleQuadratic(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
        }

class NonStationaryAltitudinalLocationQuadraticScaleQuadratic(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
        }


# Add cross terms



class NonStationaryAltitudinalScaleQuadraticCrossTermForLocation(AbstractAddCrossTermForLocation, NonStationaryAltitudinalScaleQuadratic):
    pass


class NonStationaryAltitudinalLocationLinearScaleQuadraticCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                 NonStationaryAltitudinalLocationLinearScaleQuadratic):
    pass


class NonStationaryAltitudinalLocationQuadraticScaleQuadraticCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                    NonStationaryAltitudinalLocationQuadraticScaleQuadratic):
    pass

