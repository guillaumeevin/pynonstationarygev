from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from projects.quantile_regression_vs_evt.annual_maxima_simulation.abstract_annual_maxima_simulation import \
    AnnualMaximaSimulation
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import DailyExpAnnualMaxima


class DailyExpSimulation(AnnualMaximaSimulation):

    @property
    def observations_class(self):
        return DailyExpAnnualMaxima


class StationaryExpSimulation(DailyExpSimulation):

    def create_model(self, coordinates):
        gev_param_name_to_coef_list = {
            GevParams.SCALE: [1],
        }
        return StationaryTemporalModel.from_coef_list(coordinates, gev_param_name_to_coef_list,
                                                      fit_method=TemporalMarginFitMethod.extremes_fevd_mle)


