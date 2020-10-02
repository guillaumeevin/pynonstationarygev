from extreme_fit.estimator.margin_estimator.utils import _fitted_stationary_gev
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    EurocodeConfidenceIntervalFromExtremes


def fitted_stationary_gev_with_uncertainty_interval(x_gev, fit_method=MarginFitMethod.is_mev_gev_fit,
                                                    model_class=StationaryTemporalModel,
                                                    starting_year=None,
                                                    quantile_level=0.98):
    estimator, gev_param = _fitted_stationary_gev(fit_method, model_class, starting_year, x_gev)
    if quantile_level is not None:
        EurocodeConfidenceIntervalFromExtremes.quantile_level = quantile_level
        coordinate = estimator.dataset.coordinates.df_all_coordinates.iloc[0].values
        confidence_interval = EurocodeConfidenceIntervalFromExtremes.from_estimator_extremes(estimator, ConfidenceIntervalMethodFromExtremes.ci_mle,
                                                                       coordinate)
    else:
        confidence_interval = None
    return gev_param, confidence_interval
