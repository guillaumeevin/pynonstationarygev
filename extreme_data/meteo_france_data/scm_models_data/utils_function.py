from multiprocessing import Pool

import numpy as np
from sklearn.utils import resample

from extreme_fit.estimator.margin_estimator.utils import _fitted_stationary_gev
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    EurocodeConfidenceIntervalFromExtremes
from root_utils import NB_CORES


def fitted_stationary_gev_with_uncertainty_interval(x_gev, fit_method=MarginFitMethod.is_mev_gev_fit,
                                                    model_class=StationaryTemporalModel,
                                                    starting_year=None,
                                                    quantile_level=0.98,
                                                    confidence_interval_based_on_delta_method=True):
    estimator, gev_param = _fitted_stationary_gev(fit_method, model_class, starting_year, x_gev)
    if quantile_level is not None:
        EurocodeConfidenceIntervalFromExtremes.quantile_level = quantile_level
        coordinate = estimator.dataset.coordinates.df_all_coordinates.iloc[0].values
        if confidence_interval_based_on_delta_method:
            confidence_interval = EurocodeConfidenceIntervalFromExtremes.from_estimator_extremes(estimator, ConfidenceIntervalMethodFromExtremes.ci_mle,
                                                                           coordinate)
        else:
            # Bootstrap method
            nb_bootstrap = AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP
            x_gev_list = [resample(x_gev) for _ in range(nb_bootstrap)]
            arguments = [(fit_method, model_class, starting_year, x, quantile_level) for x in x_gev_list]
            multiprocess = True
            if multiprocess:
                with Pool(NB_CORES) as p:
                    return_level_list = p.map(_compute_return_level, arguments)
            else:
                return_level_list = [_compute_return_level(argument) for argument in arguments]
            confidence_interval = tuple([np.quantile(return_level_list, q)
                                         for q in AbstractExtractEurocodeReturnLevel.bottom_and_upper_quantile])
            mean_estimate = gev_param.quantile(quantile_level)
            confidence_interval = EurocodeConfidenceIntervalFromExtremes(mean_estimate, confidence_interval)
    else:
        confidence_interval = None
    return gev_param, confidence_interval

def _compute_return_level(t):
    fit_method, model_class, starting_year, x, quantile_level = t
    return _fitted_stationary_gev(fit_method, model_class, starting_year, x)[1].quantile(quantile_level)