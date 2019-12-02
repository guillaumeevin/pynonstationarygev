from enum import Enum


class ConfidenceIntervalMethodFromExtremes(Enum):
    # Confidence interval from the ci function
    ci_bayes = 0
    ci_mle = 1
    ci_boot = 2
    ci_proflik = 3
    # Confidence interval from my functions
    my_bayes = 4


ci_method_to_method_name = {
    ConfidenceIntervalMethodFromExtremes.ci_mle: 'normal',
    ConfidenceIntervalMethodFromExtremes.ci_boot: 'boot',
    ConfidenceIntervalMethodFromExtremes.ci_proflik: 'proflik',
}

ci_method_to_color = {
    ConfidenceIntervalMethodFromExtremes.ci_mle: 'tab:brown',
    ConfidenceIntervalMethodFromExtremes.my_bayes: 'tab:green'
}