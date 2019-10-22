from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.utils import r
from extreme_fit.distribution.gev.gev_params import GevParams


class StationaryTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({})


class NonStationaryLocationTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.LOC: [self.coordinates.idx_temporal_coordinates]})

    @property
    def mul(self):
        return 1


class NonStationaryScaleTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SCALE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def sigl(self):
        return 1


class NonStationaryLogScaleTemporalModel(NonStationaryScaleTemporalModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SCALE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def siglink(self):
        return r('exp')


class NonStationaryShapeTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def shl(self):
        return 1


class NonStationaryLocationAndScaleTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.LOC: [self.coordinates.idx_temporal_coordinates],
                                       GevParams.SCALE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def mul(self):
        return 1

    @property
    def sigl(self):
        return 1