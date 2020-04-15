from extreme_fit.distribution.exp_params import ExpParams
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.utils import r
from extreme_fit.distribution.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


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


class GumbelTemporalModel(StationaryTemporalModel):

    def __init__(self, coordinates: AbstractCoordinates, use_start_value=False, params_start_fit=None,
                 params_sample=None, starting_point=None, fit_method=MarginFitMethod.is_mev_gev_fit,
                 nb_iterations_for_bayesian_fit=5000, params_start_fit_bayesian=None):
        super().__init__(coordinates, use_start_value, params_start_fit, params_sample, starting_point, fit_method,
                         nb_iterations_for_bayesian_fit, params_start_fit_bayesian, type_for_MLE="Gumbel")


class NonStationaryLocationGumbelModel(GumbelTemporalModel, NonStationaryLocationTemporalModel):
    pass


class NonStationaryScaleGumbelModel(GumbelTemporalModel, NonStationaryScaleTemporalModel):
    pass


class NonStationaryLocationAndScaleGumbelModel(GumbelTemporalModel, NonStationaryLocationAndScaleTemporalModel):
    pass
