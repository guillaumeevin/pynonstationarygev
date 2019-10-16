from extreme_estimator.extreme_models.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_estimator.extreme_models.utils import r
from extreme_estimator.margin_fits.gev.gev_params import GevParams


class StationaryStationModel(AbstractTemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({})


class NonStationaryLocationStationModel(AbstractTemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.LOC: [self.coordinates.idx_temporal_coordinates]})

    @property
    def mul(self):
        return 1


class NonStationaryScaleStationModel(AbstractTemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SCALE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def sigl(self):
        return 1


class NonStationaryLogScaleStationModel(NonStationaryScaleStationModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SCALE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def siglink(self):
        return r('exp')


class NonStationaryShapeStationModel(AbstractTemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def shl(self):
        return 1


class NonStationaryLocationAndScaleModel(AbstractTemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.LOC: [self.coordinates.idx_temporal_coordinates],
                                       GevParams.SCALE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def mul(self):
        return 1

    @property
    def sigl(self):
        return 1