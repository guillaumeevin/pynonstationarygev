from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import PolynomialMarginModel
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel


class AbstractAltitudinalModel(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function(self.param_name_to_list_dim_and_degree_for_margin_function)

    @property
    def param_name_to_list_dim_and_degree_for_margin_function(self):
        return self.param_name_to_list_dim_and_degree

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

    @property
    def param_name_to_list_dim_and_degree_for_margin_function(self):
        d = self.param_name_to_list_dim_and_degree
        assert 1 <= len(d[GevParams.LOC]) <= 2
        assert self.coordinates.idx_x_coordinates == d[GevParams.LOC][0][0]
        d[GevParams.LOC].insert(1, ((self.coordinates.idx_x_coordinates, self.coordinates.idx_temporal_coordinates), 1))
        return d


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
