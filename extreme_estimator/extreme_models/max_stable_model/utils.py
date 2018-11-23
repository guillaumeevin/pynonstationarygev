from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import \
    AbstractMaxStableModelWithCovarianceFunction, CovarianceFunction
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith, BrownResnick, Schlather, \
    Geometric, ExtremalT, ISchlather

MAX_STABLE_TYPES = [Smith, BrownResnick, Schlather, Geometric, ExtremalT, ISchlather]


def load_max_stable_models():
    # Load all max stable model
    max_stable_models = []
    for max_stable_class in MAX_STABLE_TYPES:
        if issubclass(max_stable_class, AbstractMaxStableModelWithCovarianceFunction):
            max_stable_models.extend([max_stable_class(covariance_function=covariance_function)
                                      for covariance_function in CovarianceFunction])
        else:
            max_stable_models.append(max_stable_class())
    return max_stable_models

