from enum import Enum


class ConfidenceIntervalMethodFromExtremes(Enum):
    # Confidence interval from the ci function
    ci_bayes = 0
    ci_normal = 1
    ci_boot = 2
    ci_proflik = 3
    # Confidence interval from my functions
    my_bayes = 4


ci_method_to_method_name = {
    ConfidenceIntervalMethodFromExtremes.ci_normal: 'normal',
    ConfidenceIntervalMethodFromExtremes.ci_boot: 'boot',
    ConfidenceIntervalMethodFromExtremes.ci_proflik: 'proflik',
}
