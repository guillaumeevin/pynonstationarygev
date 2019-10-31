from enum import Enum


class ConfidenceIntervalMethodFromExtremes(Enum):
    # Confidence interval from the ci function
    ci_bayes = 0
    ci_normal = 1
    ci_boot = 2
    ci_proflik = 3
    # Confidence interval from my functions
    my_bayes = 4
