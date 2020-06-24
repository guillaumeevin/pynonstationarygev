from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel


class StationaryAltitudinal(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        })


class NonStationaryAltitudinalLocationLinear(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        })


class NonStationaryAltitudinalLocationQuadratic(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        })


class NonStationaryAltitudinalLocationLinearScaleLinear(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
        })


class NonStationaryAltitudinalLocationQuadraticScaleLinear(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
        })
