from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import PolynomialMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractSpatioTemporalPolynomialModel(PolynomialMarginModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_duplicates = False


class NonStationaryLocationSpatioTemporalLinearityModel1(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [
            (self.coordinates.idx_x_coordinates, 1),
            (self.coordinates.idx_temporal_coordinates, 1),
        ]})


class NonStationaryLocationSpatioTemporalLinearityModel2(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [
            (self.coordinates.idx_x_coordinates, 1),
            (self.coordinates.idx_temporal_coordinates, 2),
        ]})


class NonStationaryLocationSpatioTemporalLinearityModel3(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [
            ((self.coordinates.idx_x_coordinates, self.coordinates.idx_temporal_coordinates), 1),
        ]})


class NonStationaryLocationSpatioTemporalLinearityModel4(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [
            (self.coordinates.idx_x_coordinates, 1),
            ((self.coordinates.idx_x_coordinates, self.coordinates.idx_temporal_coordinates), 1),
        ]})


class NonStationaryLocationSpatioTemporalLinearityModel5(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [
            ((self.coordinates.idx_x_coordinates, self.coordinates.idx_temporal_coordinates), 1),
            (self.coordinates.idx_temporal_coordinates, 1),
        ]})


class NonStationaryLocationSpatioTemporalLinearityModel6(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [
            (self.coordinates.idx_x_coordinates, 1),
            ((self.coordinates.idx_x_coordinates, self.coordinates.idx_temporal_coordinates), 1),
            (self.coordinates.idx_temporal_coordinates, 1),
        ]})


# Models that are supposed to raise errors

class NonStationaryLocationSpatioTemporalLinearityModelAssertError1(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [
            (self.coordinates.idx_temporal_coordinates, 1),
            (self.coordinates.idx_x_coordinates, 1),
        ]})


class NonStationaryLocationSpatioTemporalLinearityModelAssertError2(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [
            ((self.coordinates.idx_x_coordinates, self.coordinates.idx_temporal_coordinates), 1),
            (self.coordinates.idx_x_coordinates, 1),
        ]})


class NonStationaryLocationSpatioTemporalLinearityModelAssertError3(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [
            (self.coordinates.idx_temporal_coordinates, 1),
            ((self.coordinates.idx_x_coordinates, self.coordinates.idx_temporal_coordinates), 1),

        ]})
