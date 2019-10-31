from enum import Enum


class ConfidenceIntervalMethodFromExtremes(Enum):
    # Confidence interval from the ci function
    bayes = 0
    normal = 1
    boot = 2
    proflik = 3
    # Confidence interval from my functions
    my_bayes = 4
