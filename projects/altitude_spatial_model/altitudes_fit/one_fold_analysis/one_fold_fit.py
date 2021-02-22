import datetime
import math
import time
from itertools import chain
from multiprocessing import Pool

import numpy.testing as npt
import numpy as np
import rpy2
from cached_property import cached_property
from scipy.stats import chi2
from sklearn.utils import resample

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.distribution.gumbel.gumbel_gof import goodness_of_fit_anderson
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import AbstractMarginEstimator, \
    LinearMarginEstimator
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
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    EurocodeConfidenceIntervalFromExtremes
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import AbstractAltitudeGroup, \
    DefaultAltitudeGroup, altitudes_for_groups
from root_utils import classproperty, NB_CORES, batch
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    AnomalyTemperatureTemporalCovariate, TimeTemporalCovariate
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.slicer.split import Split
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima


class OneFoldFit(object):
    SIGNIFICANCE_LEVEL = 0.05
    best_estimator_minimizes_total_aic = False
    return_period = 100
    quantile_level = 1 - (1 / return_period)
    nb_years = 60
    last_year = 2019
    last_anomaly = 2

    def __init__(self, massif_name: str, dataset: AbstractDataset, models_classes,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 altitude_class=DefaultAltitudeGroup,
                 only_models_that_pass_goodness_of_fit_test=True,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 ):
        self.remove_physically_implausible_models = remove_physically_implausible_models
        self.confidence_interval_based_on_delta_method = confidence_interval_based_on_delta_method
        self.only_models_that_pass_goodness_of_fit_test = only_models_that_pass_goodness_of_fit_test
        self.altitude_group = altitude_class()
        self.massif_name = massif_name
        self.dataset = dataset
        self.models_classes = models_classes
        self.fit_method = fit_method
        self.temporal_covariate_for_fit = temporal_covariate_for_fit

        # Fit Estimators
        self.model_class_to_estimator = {}
        for model_class in models_classes:
            self.model_class_to_estimator[model_class] = self.fitted_linear_margin_estimator(model_class, self.dataset)
        # Compute sorted estimators indirectly
        _ = self.has_at_least_one_valid_model

        # Best estimator definition
        self.best_estimator_class_for_total_aic = None
        # Cached object
        self._folder_to_goodness = {}

    def fitted_linear_margin_estimator(self, model_class, dataset):
        return fitted_linear_margin_estimator_short(model_class=model_class,
                                                    dataset=dataset,
                                                    fit_method=self.fit_method,
                                                    temporal_covariate_for_fit=self.temporal_covariate_for_fit)

    @classproperty
    def folder_for_plots(cls):
        return 'Total aic/' if cls.best_estimator_minimizes_total_aic else ''

    @classmethod
    def get_moment_str(cls, order):
        if order == 1:
            return 'mean'
        elif order == 2:
            return 'std'
        elif order is None:
            return '{}-year return levels'.format(cls.return_period)

    def get_moment(self, altitude, temporal_covariate, order=1):
        gev_params = self.get_gev_params(altitude, temporal_covariate)
        if order == 1:
            return gev_params.mean
        elif order == 2:
            return gev_params.std
        elif order is None:
            return gev_params.return_level(return_period=self.return_period)
        else:
            raise NotImplementedError

    def get_gev_params(self, altitude, year):
        coordinate = np.array([altitude, year])
        gev_params = self.best_function_from_fit.get_params(coordinate, is_transformed=False)
        return gev_params

    def moment(self, altitudes, order=1):
        return [self.get_moment(altitude, self.covariate_after, order) for altitude in altitudes]

    @property
    def change_in_return_level_for_reference_altitude(self) -> float:
        return self.changes_of_moment(altitudes=[self.altitude_plot], order=None)[0]

    @property
    def relative_change_in_return_level_for_reference_altitude(self) -> float:
        return self.relative_changes_of_moment(altitudes=[self.altitude_plot], order=None)[0]

    def changes_of_moment(self, altitudes, order=1):
        changes = []
        for altitude in altitudes:
            mean_after = self.get_moment(altitude, self.covariate_after, order)
            mean_before = self.get_moment(altitude, self.covariate_before, order)
            change = mean_after - mean_before
            changes.append(change)
        return changes

    @property
    def covariate_before(self):
        return self._covariate_before_and_after[0]

    @property
    def covariate_after(self):
        return self._covariate_before_and_after[1]

    @property
    def _covariate_before_and_after(self):
        if self.temporal_covariate_for_fit in [None, TimeTemporalCovariate]:
            return self.last_year - self.nb_years, self.last_year
        elif self.temporal_covariate_for_fit is AnomalyTemperatureTemporalCovariate:
            # In 2020, we are roughly at 1 degree. Thus it natural to see the augmentation from 1 to 2 degree.
            return 1, self.last_anomaly
        else:
            raise NotImplementedError

    def relative_changes_of_moment(self, altitudes, order=1):
        relative_changes = []
        for altitude in altitudes:
            mean_after = self.get_moment(altitude, self.covariate_after, order)
            mean_before = self.get_moment(altitude, self.covariate_before, order)
            relative_change = 100 * (mean_after - mean_before) / mean_before
            relative_changes.append(relative_change)
        return relative_changes

    # Minimizing the AIC and some properties

    @cached_property
    def sorted_estimators(self):
        estimators = list(self.model_class_to_estimator.values())
        if self.remove_physically_implausible_models:
            # Remove wrong shape
            estimators = [e for e in estimators if -0.5 < self._compute_shape_for_reference_altitude(e) < 0.5]
            # Remove models with undefined parameters for the coordinate of interest
            coordinate = np.array([self.altitude_group.reference_altitude, self.last_year])
            estimators = [e for e in estimators if not e.function_from_fit.get_params(coordinate).has_undefined_parameters]
            if len(estimators) == 0:
                print(self.massif_name, " has only implausible models")

        sorted_estimators = sorted([estimator for estimator in estimators], key=lambda e: e.aic())
        return sorted_estimators

    def _compute_shape_for_reference_altitude(self, estimator):
        coordinate = np.array([self.altitude_plot, self.last_year])
        gev_params = estimator.function_from_fit.get_params(coordinate, is_transformed=False)
        shape = gev_params.shape
        return shape

    @cached_property
    def sorted_estimators_with_stationary(self):
        if self.only_models_that_pass_goodness_of_fit_test:
            return [e for e in self.sorted_estimators
                    if self.goodness_of_fit_test(e)
                    # and self.sensitivity_of_fit_test_top_maxima(e)
                    # and self.sensitivity_of_fit_test_last_years(e)
                    ]
        else:
            if not self.remove_physically_implausible_models:
                assert len(self.sorted_estimators) == len(self.models_classes)
            return self.sorted_estimators

    @property
    def has_at_least_one_valid_model(self):
        return len(self.sorted_estimators_with_stationary) > 0

    @property
    def model_class_to_estimator_with_finite_aic(self):
        return {type(estimator.margin_model): estimator for estimator in self.sorted_estimators}

    @property
    def best_estimator(self):
        if self.best_estimator_minimizes_total_aic and self.best_estimator_class_for_total_aic is not None:
            return self.model_class_to_estimator[self.best_estimator_class_for_total_aic]
        else:
            # With stationary
            if self.has_at_least_one_valid_model:
                best_estimator = self.sorted_estimators_with_stationary[0]
                return best_estimator
            else:
                raise ValueError('This object should not have been called because '
                                 'has_at_least_one_valid_model={}'.format(self.has_at_least_one_valid_model))

    @property
    def best_margin_model(self):
        return self.best_estimator.margin_model

    @property
    def best_function_from_fit(self):
        return self.best_estimator.function_from_fit

    @property
    def best_shape(self):
        return self.get_gev_params(altitude=self.altitude_plot, year=self.last_year).shape

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
    def model_names(self):
        return [e.margin_model.name_str for e in self.sorted_estimators]

    @property
    def best_name(self):
        name = self.best_estimator.margin_model.name_str
        latex_command = 'textbf' if self.is_significant else 'textrm'
        best_name = '$\\' + latex_command + '{' + name + '}$'
        if self.is_significant:
            best_name = '\\underline{' + best_name + '}'
        return best_name

    # Significant

    @property
    def stationary_estimator(self):
        if isinstance(self.best_estimator.margin_model, AbstractGumbelAltitudinalModel):
            return self.model_class_to_estimator_with_finite_aic[StationaryGumbelAltitudinal]
        elif isinstance(self.best_estimator.margin_model, AltitudinalOnlyScale):
            return self.model_class_to_estimator_with_finite_aic[StationaryAltitudinalOnlyScale]
        elif isinstance(self.best_estimator.margin_model, AltitudinalShapeLinearTimeStationary):
            return self.model_class_to_estimator_with_finite_aic[AltitudinalShapeLinearTimeStationary]
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

    @cached_property
    def is_significant(self) -> bool:
        if self.confidence_interval_based_on_delta_method:
            stationary_model_classes = [StationaryAltitudinal, StationaryGumbelAltitudinal,
                                        AltitudinalShapeLinearTimeStationary]
            if any([isinstance(self.best_estimator.margin_model, c)
                    for c in stationary_model_classes]):
                return False
            else:
                return self.likelihood_ratio > chi2.ppf(q=1 - self.SIGNIFICANCE_LEVEL, df=self.degree_freedom_chi2)
        else:
            # Bootstrap based significance
            return self.cached_results_from_bootstrap[0]

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

    def sensitivity_of_fit_test_top_maxima(self, estimator: LinearMarginEstimator):
        # Build the dataset without the maxima for each altitude
        new_dataset = AbstractDataset.remove_top_maxima(self.dataset.observations,
                                                        self.dataset.coordinates)
        # Fit the new estimator
        new_estimator = fitted_linear_margin_estimator_short(model_class=type(estimator.margin_model),
                                                             dataset=new_dataset,
                                                             fit_method=self.fit_method,
                                                             temporal_covariate_for_fit=self.temporal_covariate_for_fit)
        # Compare sign of change
        has_not_opposite_sign = self.sign_of_change(estimator.function_from_fit) * self.sign_of_change(
            new_estimator.function_from_fit) >= 0
        return has_not_opposite_sign

    def sensitivity_of_fit_test_last_years(self, estimator: LinearMarginEstimator):
        # Build the dataset without the maxima for each altitude
        new_dataset = AbstractDataset.remove_last_maxima(self.dataset.observations,
                                                         self.dataset.coordinates,
                                                         nb_years=10)
        # Fit the new estimator
        model_class = type(estimator.margin_model)
        new_estimator = fitted_linear_margin_estimator_short(model_class=model_class,
                                                             dataset=new_dataset,
                                                             fit_method=self.fit_method,
                                                             temporal_covariate_for_fit=self.temporal_covariate_for_fit)
        # Compare sign of change
        has_not_opposite_sign = self.sign_of_change(estimator.function_from_fit) * self.sign_of_change(
            new_estimator.function_from_fit) >= 0
        # if not has_not_opposite_sign:
        # print('Last years', self.massif_name, model_class, self.sign_of_change(estimator), self.sign_of_change(new_estimator))
        return has_not_opposite_sign

    def sign_of_change(self, function_from_fit):
        return_levels = []
        for temporal_covariate in self._covariate_before_and_after:
            coordinate = np.array([self.altitude_plot, temporal_covariate])
            return_level = function_from_fit.get_params(
                coordinate=coordinate,
                is_transformed=False).return_level(return_period=self.return_period)
            return_levels.append(return_level)
        return 100 * (return_levels[1] - return_levels[0]) / return_levels[0]

    def goodness_of_fit_test(self, estimator):
        quantiles = estimator.sorted_empirical_standard_gumbel_quantiles()
        goodness_of_fit_anderson_test = goodness_of_fit_anderson(quantiles, self.SIGNIFICANCE_LEVEL)
        return goodness_of_fit_anderson_test

    def standard_gumbel_quantiles(self, n=None):
        standard_gumbel_distribution = GevParams(loc=0, scale=1, shape=0)
        if n is None:
            n = len(self.dataset.coordinates)
        standard_gumbel_quantiles = [standard_gumbel_distribution.quantile(i / (n + 1)) for i in range(1, n + 1)]
        return standard_gumbel_quantiles

    def best_confidence_interval(self, altitude, year) -> EurocodeConfidenceIntervalFromExtremes:
        coordinate = np.array([altitude, year])
        if self.confidence_interval_based_on_delta_method:
            EurocodeConfidenceIntervalFromExtremes.quantile_level = self.quantile_level
            return EurocodeConfidenceIntervalFromExtremes.from_estimator_extremes(
                estimator_extremes=self.best_estimator,
                ci_method=ConfidenceIntervalMethodFromExtremes.ci_mle,
                coordinate=coordinate)
        else:
            key = (altitude, year)
            mean_estimate = self.cached_results_from_bootstrap[1][key]
            confidence_interval = self.cached_results_from_bootstrap[2][key]
            return EurocodeConfidenceIntervalFromExtremes(mean_estimate, confidence_interval)

    def get_return_level(self, function_from_fit, coordinate):
        return function_from_fit.get_params(coordinate).return_level(self.return_period)

    @cached_property
    def best_residuals(self):
        return self.best_estimator.sorted_empirical_standard_gumbel_quantiles(split=Split.all)

    # @property
    # def bootstrap_data(self):
    #     start = time.time()
    #     bootstrap = []
    #     for _ in range(AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP):
    #         residuals = self.best_estimator.sorted_empirical_standard_gumbel_quantiles(split=Split.all)
    #
    #         # yield coordinate_values_to_maxima
    #         bootstrap.append(coordinate_values_to_maxima)
    #     end1 = time.time()
    #     duration = str(datetime.timedelta(seconds=end1 - start))
    #     print('bootstrap loader duration', duration)
    #     return bootstrap

    # def bootstrap_batch_data(self, batchsize=20):
    #     bootstrap_batch_data = []
    #     batch = []
    #     len_batch = 0
    #     for bootstrap in self.bootstrap_data:
    #         batch.append(bootstrap)
    #         len_batch += 1
    #         if len_batch == batchsize:
    #             yield batch
    #             batch = []
    #             len_batch = 0
    #     return bootstrap_batch_data

    @cached_property
    def cached_results_from_bootstrap(self):
        start = time.time()
        bootstrap_fitted_functions = self.bootstrap_fitted_functions_from_fit
        end1 = time.time()
        duration = str(datetime.timedelta(seconds=end1 - start))
        print('Fit duration', duration)

        # First result - Compute the significance
        sign_of_changes = [self.sign_of_change(f) for f in bootstrap_fitted_functions]
        if self.sign_of_change(self.best_function_from_fit) > 0:
            is_significant = np.quantile(sign_of_changes, self.SIGNIFICANCE_LEVEL) > 0
        else:
            is_significant = np.quantile(sign_of_changes, 1 - self.SIGNIFICANCE_LEVEL) < 0

        # Second result - Compute some dictionary for the return level
        altitude_and_year_to_return_level_mean_estimate = {}
        altitude_and_year_to_return_level_confidence_interval = {}
        altitudes = altitudes_for_groups[self.altitude_group.group_id - 1]
        years = [1959, 2019]
        for year in years:
            for altitude in altitudes:
                key = (altitude, year)
                coordinate = np.array([altitude, year])
                mean_estimate = self.get_return_level(self.best_function_from_fit, coordinate)
                bootstrap_return_levels = [self.get_return_level(f, coordinate) for f in
                                           bootstrap_fitted_functions]
                confidence_interval = tuple([np.quantile(bootstrap_return_levels, q)
                                             for q in AbstractExtractEurocodeReturnLevel.bottom_and_upper_quantile])
                altitude_and_year_to_return_level_mean_estimate[key] = mean_estimate
                altitude_and_year_to_return_level_confidence_interval[key] = confidence_interval

        return is_significant, altitude_and_year_to_return_level_mean_estimate, altitude_and_year_to_return_level_confidence_interval


    @property
    def bootstrap_fitted_functions_from_fit(self):
        print('nb of bootstrap for confidence interval=', AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP)
        multiprocess = None
        idxs = list(range(AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP))

        if multiprocess is None:
            start = time.time()
            with Pool(NB_CORES) as p:
                batchsize = math.ceil(AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP / NB_CORES)
                list_functions_from_fit = p.map(self.fit_batch_bootstrap_estimator, batch(idxs, batchsize=batchsize))
                functions_from_fit = list(chain.from_iterable(list_functions_from_fit))

        elif multiprocess:
            print('multiprocessing')
            start = time.time()
            with Pool(NB_CORES) as p:
                functions_from_fit = p.map(self.fit_one_bootstrap_estimator, idxs)

        else:
            start = time.time()

            functions_from_fit = []
            for idx in idxs:
                estimator = self.fit_one_bootstrap_estimator(idx)
                functions_from_fit.append(estimator)

        end1 = time.time()
        duration = str(datetime.timedelta(seconds=end1 - start))
        print('Multiprocessing duration', duration)
        return functions_from_fit

    def fit_batch_bootstrap_estimator(self, idxs):
        list_function_from_fit = []
        for idx in idxs:
            list_function_from_fit.append(self.fit_one_bootstrap_estimator(idx))
        return list_function_from_fit

    def fit_one_bootstrap_estimator(self, idx):
        resample_residuals = resample(self.best_residuals)
        coordinate_values_to_maxima = self.best_estimator. \
            coordinate_values_to_maxima_from_standard_gumbel_quantiles(standard_gumbel_quantiles=resample_residuals)

        coordinates = self.dataset.coordinates
        observations = AnnualMaxima.from_coordinates(coordinates, coordinate_values_to_maxima)
        dataset = AbstractDataset(observations=observations, coordinates=coordinates)
        model_class = type(self.best_margin_model)

        function_from_fit = self.fitted_linear_margin_estimator(model_class, dataset).function_from_fit

        return function_from_fit
