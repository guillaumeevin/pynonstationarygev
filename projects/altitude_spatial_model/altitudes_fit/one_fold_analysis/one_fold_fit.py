import numpy.testing as npt
import numpy as np
import rpy2
from cached_property import cached_property
from scipy.stats import chi2

from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator_short
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal
from extreme_fit.model.margin_model.polynomial_margin_model.gumbel_altitudinal_models import \
    StationaryGumbelAltitudinal, AbstractGumbelAltitudinalModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from root_utils import classproperty


class OneFoldFit(object):
    SIGNIFICANCE_LEVEL = 0.05
    best_estimator_minimizes_mean_aic = False

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

        # Best estimator definition
        self.best_estimator_class_for_mean_aic = None

    @classproperty
    def folder_for_plots(cls):
        return 'Mean aic' if cls.best_estimator_minimizes_mean_aic else 'Individual aic'

    @classmethod
    def get_moment_str(cls, order):
        if order == 1:
            return 'Mean'
        elif order == 2:
            return 'Std'
        elif order is None:
            return 'Return level'

    def get_moment(self, altitude, year, order=1):
        gev_params = self.get_gev_params(altitude, year)
        if order == 1:
            return gev_params.mean
        elif order == 2:
            return gev_params.std
        elif order is None:
            return gev_params.return_level(return_period=2019)
        else:
            raise NotImplementedError

    def get_gev_params(self, altitude, year):
        coordinate = np.array([altitude, year])
        gev_params = self.best_function_from_fit.get_params(coordinate, is_transformed=False)
        return gev_params

    def moment(self, altitudes, year=2019, order=1):
        return [self.get_moment(altitude, year, order) for altitude in altitudes]

    def changes_in_the_moment(self, altitudes, year=2019, nb_years=50, order=1):
        changes = []
        for altitude in altitudes:
            mean_after = self.get_moment(altitude, year, order)
            mean_before = self.get_moment(altitude, year - nb_years, order)
            change = mean_after - mean_before
            changes.append(change)
        return changes

    def relative_changes_in_the_moment(self, altitudes, year=2019, nb_years=50, order=1):
        relative_changes = []
        for altitude in altitudes:
            mean_after = self.get_moment(altitude, year, order)
            mean_before = self.get_moment(altitude, year - nb_years, order)
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

    @property
    def model_class_to_estimator_with_finite_aic(self):
        return {type(estimator.margin_model): estimator for estimator in self.sorted_estimators_with_finite_aic}

    @cached_property
    def set_estimators_with_finite_aic(self):
        return set(self.sorted_estimators_with_finite_aic)

    @property
    def best_estimator(self):
        if self.best_estimator_minimizes_mean_aic and self.best_estimator_class_for_mean_aic is not None:
            return self.model_class_to_estimator[self.best_estimator_class_for_mean_aic]
        else:
            return self.sorted_estimators_with_finite_aic[0]

    @property
    def best_estimator_has_finite_aic(self):
        return self.best_estimator in self.set_estimators_with_finite_aic

    @property
    def best_shape(self):
        # We take any altitude (altitude=1000 for instance) as the shape is constant w.r.t the altitude
        return self.get_gev_params(altitude=1000, year=2019).shape

    @property
    def best_function_from_fit(self):
        return self.best_estimator.function_from_fit

    @property
    def best_name(self):
        name = self.best_estimator.margin_model.name_str
        latex_command = 'textbf' if self.is_significant else 'textrm'
        return '$\\' + latex_command + '{' + name + '}$'

    # Significant

    @property
    def stationary_estimator(self):
        if isinstance(self.best_estimator.margin_model, AbstractGumbelAltitudinalModel):
            return self.model_class_to_estimator_with_finite_aic[StationaryGumbelAltitudinal]
        else:
            return self.model_class_to_estimator_with_finite_aic[StationaryAltitudinal]

    @property
    def likelihood_ratio(self):
        return self.stationary_estimator.deviance() - self.best_estimator.deviance()

    @property
    def degree_freedom_chi2(self):
        return self.best_estimator.margin_model.nb_params - self.stationary_estimator.margin_model.nb_params

    @property
    def is_significant(self) -> bool:
        stationary_model_classes = [StationaryAltitudinal, StationaryGumbelAltitudinal]
        if any([isinstance(self.best_estimator.margin_model, c) 
                for c in stationary_model_classes]):
            return False
        else:
            return self.likelihood_ratio > chi2.ppf(q=1 - self.SIGNIFICANCE_LEVEL, df=self.degree_freedom_chi2)
