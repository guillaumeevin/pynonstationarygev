import numpy.testing as npt
import numpy as np
import rpy2
from cached_property import cached_property

from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator_short
from extreme_fit.model.margin_model.utils import MarginFitMethod


class OneFoldFit(object):

    def __init__(self, massif_name, dataset, models_classes, fit_method=MarginFitMethod.extremes_fevd_mle):
        self.massif_name = massif_name
        self.dataset = dataset
        self.models_classes = models_classes
        self.fit_method = fit_method

        # Fit Estimators
        self.model_class_to_estimator = {}
        for model_class in models_classes:
            self.model_class_to_estimator[model_class] = fitted_linear_margin_estimator_short(model_class=model_class,
                                                                                              dataset=self.dataset,
                                                                                              fit_method=self.fit_method)

    def mean(self, altitudes, year=2019):
        return [self.get_mean(altitude, year) for altitude in altitudes]

    def get_mean(self, altitude, year):
        return self.get_gev_params(altitude, year).mean

    def get_gev_params(self, altitude, year):
        coordinate = np.array([altitude, year])
        gev_params = self.best_function_from_fit.get_params(coordinate, is_transformed=False)
        return gev_params

    def relative_changes_in_the_mean(self, altitudes, year=2019, nb_years=50):
        relative_changes = []
        for altitude in altitudes:
            mean_after = self.get_mean(altitude, year)
            mean_before = self.get_mean(altitude, year - nb_years)
            relative_change = 100 * (mean_after - mean_before) / mean_before
            relative_changes.append(relative_change)
        return relative_changes

    # Minimizing the AIC and some properties

    @cached_property
    def sorted_estimators_with_finite_aic(self):
        estimators = list(self.model_class_to_estimator.values())
        estimators_with_finite_aic = []
        for estimator in estimators:
            try:
                aic = estimator.aic()
                npt.assert_almost_equal(estimator.result_from_model_fit.aic, aic, decimal=5)
                print(self.massif_name, estimator.margin_model.name_str, aic)
                estimators_with_finite_aic.append(estimator)
            except (AssertionError, rpy2.rinterface.RRuntimeError):
                print(self.massif_name, estimator.margin_model.name_str, 'infinite aic')
        print('Summary {}: {}/{} fitted'.format(self.massif_name, len(estimators_with_finite_aic), len(estimators)))
        sorted_estimators_with_finite_aic = sorted([estimator for estimator in estimators_with_finite_aic],
                                                   key=lambda e: e.aic())
        return sorted_estimators_with_finite_aic

    @cached_property
    def best_estimator(self):
        return self.sorted_estimators_with_finite_aic[0]

    @property
    def best_shape(self):
        return self.get_gev_params(1000, year=2019).shape

    @property
    def best_function_from_fit(self):
        return self.best_estimator.function_from_fit

    @property
    def best_name(self):
        return self.best_estimator.margin_model.name_str
