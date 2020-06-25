from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel


class AbstractAltitudinalModel(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function(self.param_name_to_list_dim_and_degree)

    @property
    def param_name_to_list_dim_and_degree(self):
        raise NotImplementedError


class StationaryAltitudinal(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        }


class NonStationaryAltitudinalLocationLinear(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        }


class NonStationaryAltitudinalLocationQuadratic(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        }


class NonStationaryAltitudinalLocationLinearScaleLinear(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
        }


class NonStationaryAltitudinalLocationQuadraticScaleLinear(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
        }


# Add cross terms

class AbstractAddCrossTermForLocation(AbstractAltitudinalModel):

    def load_margin_function(self, param_name_to_dims=None):
        d = self.param_name_to_list_dim_and_degree
        d[GevParams.LOC] += ((self.coordinates.idx_x_coordinates, self.coordinates.idx_temporal_coordinates), 1)
        return super().load_margin_function(d)


class NonStationaryCrossTermForLocation(AbstractAddCrossTermForLocation, StationaryAltitudinal):
    pass


class NonStationaryAltitudinalLocationLinearCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                 NonStationaryAltitudinalLocationLinear):
    pass


class NonStationaryAltitudinalLocationQuadraticCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                    NonStationaryAltitudinalLocationQuadratic):
    pass


class NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                            NonStationaryAltitudinalLocationLinearScaleLinear):
    pass


class NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                               NonStationaryAltitudinalLocationQuadraticScaleLinear):
    pass
