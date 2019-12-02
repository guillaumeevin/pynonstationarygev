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
    ConfidenceIntervalMethodFromExtremes.ci_mle: 'mediumseagreen',
    ConfidenceIntervalMethodFromExtremes.my_bayes: 'darkgreen'
}

# common_part_and_uncertainty = 'Return level and its uncertainty with the'
ci_method_to_label = {
    ConfidenceIntervalMethodFromExtremes.ci_mle: 'Delta method',
    ConfidenceIntervalMethodFromExtremes.my_bayes: 'Bayesian procedure'
}