import numpy as np
from cached_property import cached_property

from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator_short
from extreme_fit.model.margin_model.utils import MarginFitMethod


class OneFoldFit(object):

    def __init__(self, dataset, models_classes, fit_method=MarginFitMethod.extremes_fevd_mle):
        self.dataset = dataset
        self.models_classes = models_classes
        self.fit_method = fit_method

        # Fit Estimators
        self.model_class_to_estimator = {}
        for model_class in models_classes:
            self.model_class_to_estimator[model_class] = fitted_linear_margin_estimator_short(model_class=model_class,
                                                                                              dataset=self.dataset,
                                                                                              fit_method=self.fit_method)
        # Some display
        for estimator in self.model_class_to_estimator.values():
            print(estimator.result_from_model_fit.aic)

    @cached_property
    def best_estimator(self):
        sorted_estimators = sorted([estimator for estimator in self.model_class_to_estimator.values()],
                                   key=lambda e: e.result_from_model_fit.aic)
        estimator_that_minimizes_aic = sorted_estimators[0]
        return estimator_that_minimizes_aic

    @property
    def best_function_from_fit(self):
        return self.best_estimator.function_from_fit

    def mean(self, altitudes, year=2019):
        return [self.get_mean(altitude, year) for altitude in altitudes]

    def get_mean(self, altitude, year):
        coordinate = np.array([altitude, year])
        gev_params = self.best_function_from_fit.get_params(coordinate, is_transformed=False)
        return gev_params.mean

    def relative_changes_in_the_mean(self, altitudes, year=2019, nb_years=50):
        relative_changes = []
        for altitude in altitudes:
            mean_after = self.get_mean(altitude, year)
            mean_before = self.get_mean(altitude, year - nb_years)
            relative_change = 100 * (mean_after - mean_before) / mean_before
            relative_changes.append(relative_change)
        return relative_changes




