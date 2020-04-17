from enum import Enum


class MarginFitMethod(Enum):
    is_mev_gev_fit = 0
    extremes_fevd_bayesian = 1
    extremes_fevd_mle = 2
    extremes_fevd_gmle = 3
    extremes_fevd_l_moments = 4
    spatial_extremes_mle = 5


FEVD_MARGIN_FIT_METHOD_TO_ARGUMENT_STR = {
    MarginFitMethod.extremes_fevd_mle: "MLE",
    MarginFitMethod.extremes_fevd_gmle: "GMLE",
    MarginFitMethod.extremes_fevd_l_moments: "Lmoments",
    MarginFitMethod.extremes_fevd_bayesian: "Bayesian"
}
FEVD_MARGIN_FIT_METHODS = set(FEVD_MARGIN_FIT_METHOD_TO_ARGUMENT_STR.keys())

def fitmethod_to_str(fit_method):
    return str(fit_method).split('.')[-1]
