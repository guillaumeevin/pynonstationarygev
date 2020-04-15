from enum import Enum


class MarginFitMethod(Enum):
    is_mev_gev_fit = 0
    extremes_fevd_bayesian = 1
    extremes_fevd_mle = 2
    extremes_fevd_gmle = 3
    extremes_fevd_l_moments = 4

def fitmethod_to_str(fit_method):
    return str(fit_method).split('.')[-1]
