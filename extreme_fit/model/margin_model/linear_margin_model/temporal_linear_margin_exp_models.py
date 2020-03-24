from extreme_fit.distribution.exp_params import ExpParams
from extreme_fit.model.daily_data_model import AbstractModelOnDailyData
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel


class NonStationaryRateTemporalModel(AbstractTemporalLinearMarginModel, AbstractModelOnDailyData):

    def __init__(self, *arg, **kwargs):
        kwargs['params_class'] = ExpParams
        super().__init__(*arg, **kwargs)

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({ExpParams.RATE: [self.coordinates.idx_temporal_coordinates]})
