from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.spline_margin_model.spline_margin_model import SplineMarginModel


class NonStationaryTwoLinearLocationModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        # Degree 1, Two Linear sections for the location
        return super().load_margin_function({GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1, 2)]})


class NonStationaryTwoLinearScaleModel(SplineMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        # Degree 1, Two Linear sections for the scale parameters
        return super().load_margin_function({GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 1, 2)]})
