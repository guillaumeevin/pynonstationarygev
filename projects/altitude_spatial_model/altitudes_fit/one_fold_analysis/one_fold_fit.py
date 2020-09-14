import numpy.testing as npt
import numpy as np
import rpy2
from cached_property import cached_property
from scipy.stats import chi2

from extreme_fit.distribution.gumbel.gumbel_gof import goodness_of_fit_anderson
from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator_short
from extreme_fit.function.param_function.polynomial_coef import PolynomialAllCoef, PolynomialCoef
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models_only_altitude_and_scale import \
    AltitudinalOnlyScale, StationaryAltitudinalOnlyScale
from extreme_fit.model.margin_model.polynomial_margin_model.gumbel_altitudinal_models import \
    StationaryGumbelAltitudinal, AbstractGumbelAltitudinalModel
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_linear_shape_wrt_altitude import \
    AltitudinalShapeLinearTimeStationary
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import AbstractAltitudeGroup, \
    DefaultAltitudeGroup
from root_utils import classproperty
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class OneFoldFit(object):
    SIGNIFICANCE_LEVEL = 0.05
    best_estimator_minimizes_total_aic = False

    def __init__(self, massif_name: str, dataset: AbstractDataset, models_classes,
                 fit_method=MarginFitMethod.extremes_fevd_mle, temporal_covariate_for_fit=None,
                 altitude_class=DefaultAltitudeGroup,
                 only_models_that_pass_anderson_test=True):
        self.only_models_that_pass_anderson_test = only_models_that_pass_anderson_test
        self.altitude_group = altitude_class()
        self.massif_name = massif_name
        self.dataset = dataset
        self.models_classes = models_classes
        self.fit_method = fit_method
        self.temporal_covariate_for_fit = temporal_covariate_for_fit

        # Fit Estimators
        self.model_class_to_estimator = {}
        for model_class in models_classes:
            self.model_class_to_estimator[model_class] = fitted_linear_margin_estimator_short(model_class=model_class,
                                                                                              dataset=self.dataset,
                                                                                              fit_method=self.fit_method,
                                                                                              temporal_covariate_for_fit=self.temporal_covariate_for_fit)

        # Best estimator definition
        self.best_estimator_class_for_total_aic = None
        # Cached object
        self._folder_to_goodness = {}

    @classproperty
    def folder_for_plots(cls):
        return 'Total aic/' if cls.best_estimator_minimizes_total_aic else ''

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
            return gev_params.return_level(return_period=50)
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
    def sorted_estimators(self):
        estimators = list(self.model_class_to_estimator.values())
        sorted_estimators = sorted([estimator for estimator in estimators], key=lambda e: e.aic())
        return sorted_estimators

    @cached_property
    def _sorted_estimators_without_stationary(self):
        return [e for e in self.sorted_estimators if not isinstance(e.margin_model, StationaryAltitudinal)]

    @cached_property
    def sorted_estimators_without_stationary(self):
        if self.only_models_that_pass_anderson_test:
            return [e for e in self._sorted_estimators_without_stationary if self.goodness_of_fit_test(e)]
        else:
            return self._sorted_estimators_without_stationary

    @property
    def has_at_least_one_valid_non_stationary_model(self):
        return len(self.sorted_estimators_without_stationary) > 0

    @property
    def model_class_to_estimator_with_finite_aic(self):
        return {type(estimator.margin_model): estimator for estimator in self.sorted_estimators}

    @property
    def best_estimator(self):
        if self.best_estimator_minimizes_total_aic and self.best_estimator_class_for_total_aic is not None:
            return self.model_class_to_estimator[self.best_estimator_class_for_total_aic]
        else:
            if self.has_at_least_one_valid_non_stationary_model:
                best_estimator = self.sorted_estimators_without_stationary[0]
                return best_estimator
            else:
                raise ValueError('This should not happen')

    @property
    def best_margin_model(self):
        return self.best_estimator.margin_model

    @property
    def best_function_from_fit(self):
        return self.best_estimator.function_from_fit

    @property
    def best_shape(self):
        return self.get_gev_params(altitude=self.altitude_plot, year=2019).shape

    @property
    def altitude_plot(self):
        return self.altitude_group.reference_altitude

    def best_coef(self, param_name, dim, degree):
        try:
            coef = self.best_function_from_fit.param_name_to_coef[param_name]  # type: PolynomialAllCoef
            coef = coef.dim_to_polynomial_coef[dim]  # type: PolynomialCoef
            coef = coef.idx_to_coef[degree]
            return coef
        except (TypeError, KeyError):
            return None

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
        elif isinstance(self.best_estimator.margin_model, AltitudinalOnlyScale):
            return self.model_class_to_estimator_with_finite_aic[StationaryAltitudinalOnlyScale]
        elif isinstance(self.best_estimator.margin_model, AltitudinalShapeLinearTimeStationary):
            return self.model_class_to_estimator_with_finite_aic[AltitudinalShapeLinearTimeStationary]
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

    # @property
    # def goodness_of_fit_anderson_test(self):
    #     if self.folder_for_plots in self._folder_to_goodness:
    #         return self._folder_to_goodness[self.folder_for_plots]
    #     else:
    #         estimator = self.best_estimator
    #         goodness_of_fit_anderson_test = self.goodness_of_fit_test(estimator)
    #         if not goodness_of_fit_anderson_test:
    #             print('{} with {} does not pass the anderson test for model {}'.format(self.massif_name,
    #                                                                                    self.folder_for_plots,
    #                                                                                    type(self.best_margin_model)))
    #         self._folder_to_goodness[self.folder_for_plots] = goodness_of_fit_anderson_test
    #         return goodness_of_fit_anderson_test

    def goodness_of_fit_test(self, estimator):
        quantiles = self.compute_empirical_quantiles(estimator=estimator)
        goodness_of_fit_anderson_test = goodness_of_fit_anderson(quantiles, self.SIGNIFICANCE_LEVEL)
        return goodness_of_fit_anderson_test

    def compute_empirical_quantiles(self, estimator):
        empirical_quantiles = []
        df_maxima_gev = self.dataset.observations.df_maxima_gev
        df_coordinates = self.dataset.coordinates.df_coordinates()
        for coordinate, maximum in zip(df_coordinates.values.copy(), df_maxima_gev.values.copy()):
            gev_param = estimator.function_from_fit.get_params(
                coordinate=coordinate,
                is_transformed=False)
            assert len(maximum) == 1
            maximum_standardized = gev_param.gumbel_standardization(maximum[0])
            empirical_quantiles.append(maximum_standardized)
        empirical_quantiles = sorted(empirical_quantiles)
        return empirical_quantiles
