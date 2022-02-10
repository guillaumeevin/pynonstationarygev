import datetime
import math
import time
from itertools import chain
from multiprocessing import Pool
from typing import List

import numpy as np
from cached_property import cached_property
from scipy.stats import chi2
from sklearn.utils import resample

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.distribution.gumbel.gumbel_gof import goodness_of_fit_anderson
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator_short
from extreme_fit.function.margin_function.independent_margin_function import IndependentMarginFunction
from extreme_fit.function.param_function.polynomial_coef import PolynomialAllCoef, PolynomialCoef
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models_only_altitude_and_scale import \
    AltitudinalOnlyScale, StationaryAltitudinalOnlyScale
from extreme_fit.model.margin_model.polynomial_margin_model.gumbel_altitudinal_models import \
    StationaryGumbelAltitudinal, AbstractGumbelAltitudinalModel
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_linear_shape_wrt_altitude import \
    AltitudinalShapeLinearTimeStationary
from extreme_fit.model.margin_model.spline_margin_model.spline_margin_model import SplineMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    EurocodeConfidenceIntervalFromExtremes
from extreme_fit.model.utils import SafeRunException
from extreme_trend.one_fold_fit.altitude_group import DefaultAltitudeGroup, altitudes_for_groups
from extreme_trend.one_fold_fit.utils_split_sample_selection import compute_mean_log_score_with_split_sample
from projected_extremes.results.combination_utils import load_combination, generate_sub_combination, \
    load_param_name_to_climate_coordinates_with_effects
from root_utils import NB_CORES, batch, get_display_name_from_object_type
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima


class OneFoldFit(object):
    SIGNIFICANCE_LEVEL = 0.05
    return_period = 100
    quantile_level = 1 - (1 / return_period)
    multiprocessing = None
    nb_cores_for_multiprocess = NB_CORES
    max_batchsize = None
    SELECTION_METHOD_NAME = 'aic'
    COVARIATE_BEFORE_TEMPERATURE = 1
    COVARIATE_AFTER_TEMPERATURE = 2
    COVARIATE_BEFORE_DISPLAY = None
    COVARIATE_AFTER_DISPLAY = None

    def __init__(self, massif_name: str, dataset: AbstractDataset, models_classes,
                 first_year=None, last_year=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 altitude_group=None,
                 only_models_that_pass_goodness_of_fit_test=True,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 param_name_to_climate_coordinates_with_effects=None,
                 linear_effects=(False, False, False),
                 with_sub_combinations=False):
        self.linear_effects = linear_effects
        self.with_sub_combinations = with_sub_combinations
        self.first_year = first_year
        self.last_year = last_year
        self.remove_physically_implausible_models = remove_physically_implausible_models
        self.confidence_interval_based_on_delta_method = confidence_interval_based_on_delta_method
        self.only_models_that_pass_goodness_of_fit_test = only_models_that_pass_goodness_of_fit_test
        self.altitude_group = altitude_group
        self.massif_name = massif_name
        self.dataset = dataset
        self.models_classes = models_classes
        self.fit_method = fit_method
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.param_name_to_climate_coordinates_with_effects = param_name_to_climate_coordinates_with_effects
        # Load combinations
        combination = load_combination(self.param_name_to_climate_coordinates_with_effects)
        if self.with_sub_combinations:
            self.sub_combinations = generate_sub_combination(combination)
        else:
            self.sub_combinations = [combination]
        # Fit Estimators
        self.fitted_estimators = set()
        for model_class in models_classes:
            for sub_combination in self.sub_combinations:
                param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                    sub_combination)
                fitted_estimator = self.fitted_linear_margin_estimator(model_class, self.dataset,
                                                                       param_name_to_climate_coordinates_with_effects)
                self.fitted_estimators.add(fitted_estimator)
        # Compute sorted estimators indirectly
        _ = self.has_at_least_one_valid_model

    def fitted_linear_margin_estimator(self, model_class, dataset, param_name_to_climate_coordinates_with_effects):
        if issubclass(model_class, SplineMarginModel):
            assert self.fit_method is MarginFitMethod.evgam
        estimator = fitted_linear_margin_estimator_short(model_class=model_class, dataset=dataset,
                                                         fit_method=self.fit_method,
                                                         temporal_covariate_for_fit=self.temporal_covariate_for_fit,
                                                         drop_duplicates=False,
                                                         param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                                         linear_effects=self.linear_effects)
        # assert that is not inf
        assert not np.isinf(estimator.nllh)
        return estimator

    @classmethod
    def get_moment_str(cls, order):
        if order == 1:
            return 'mean'
        elif order == 2:
            return 'std'
        elif order is None:
            return '{}-year return levels'.format(cls.return_period)

    def get_moment_for_plots(self, altitudes, order=1, covariate_before=None, covariate_after=None):
        return [self.get_moment(altitudes[0], covariate_after, order)]

    def get_moment_covariate_float_or_list(self, altitude, temporal_covariate, order=1):
        if isinstance(temporal_covariate, tuple):
            return np.mean([self.get_moment(altitude, c, order) for c in temporal_covariate])
        else:
            return self.get_moment(altitude, temporal_covariate, order)

    def get_moment(self, altitude, temporal_covariate, order=1):
        gev_params = self.get_gev_params(altitude, temporal_covariate)
        if order == 1:
            return gev_params.mean
        elif order == 2:
            return gev_params.std
        elif order is None:
            return gev_params.return_level(return_period=self.return_period)
        elif order in GevParams.PARAM_NAMES:
            d = gev_params.to_dict()
            assert isinstance(d, dict)
            return d[order]
        else:
            raise NotImplementedError

    def get_gev_params(self, altitude, year):
        coordinate = self.get_coordinate(altitude, year)
        gev_params = self.best_margin_function_from_fit.get_params(coordinate, is_transformed=False)
        return gev_params

    def moment(self, altitudes, order=1):
        return [self.get_moment(altitude, self.covariate_after, order) for altitude in altitudes]

    @property
    def change_in_return_level_for_reference_altitude(self) -> float:
        return self.changes_of_moment(altitudes=[self.altitude_plot], order=None)[0]

    @property
    def relative_change_in_return_level_for_reference_altitude(self) -> float:
        return self.relative_changes_of_moment(altitudes=[self.altitude_plot], order=None)[0]

    @property
    def relative_change_in_mean_for_reference_altitude(self) -> float:
        return self.relative_changes_of_moment(altitudes=[self.altitude_plot], order=1)[0]

    @property
    def change_in_mean_for_reference_altitude(self) -> float:
        return self.changes_of_moment(altitudes=[self.altitude_plot], order=1)[0]


    def changes_of_moment(self, altitudes, order=1, covariate_before=None, covariate_after=None):
        covariate_after, covariate_before = self.set_covariate_before_and_after(covariate_after, covariate_before)
        changes = []
        for altitude in altitudes:
            mean_after = self.get_moment_covariate_float_or_list(altitude, covariate_after, order)
            mean_before = self.get_moment_covariate_float_or_list(altitude, covariate_before, order)
            change = mean_after - mean_before
            changes.append(change)
        return changes

    def set_covariate_before_and_after(self, covariate_after, covariate_before):
        if covariate_before is None:
            covariate_before = self.covariate_before
        if covariate_after is None:
            covariate_after = self.covariate_after
        return covariate_after, covariate_before

    @property
    def covariate_before(self):
        return self._covariate_before_and_after[0]

    @property
    def covariate_after(self):
        return self._covariate_before_and_after[1]

    @property
    def _covariate_before_and_after(self):
        if self.temporal_covariate_for_fit in [None, TimeTemporalCovariate]:
            return self.first_year, self.last_year
        elif self.temporal_covariate_for_fit is AnomalyTemperatureWithSplineTemporalCovariate:
            # In 2020, we are roughly at 1 degree. Thus it natural to see the augmentation from 1 to 2 degree.
            return self.COVARIATE_BEFORE_TEMPERATURE, self.COVARIATE_AFTER_TEMPERATURE
        else:
            raise NotImplementedError

    @property
    def between_covariate_str(self):
        d_temperature = {'C': '{C}'}
        if self.COVARIATE_BEFORE_DISPLAY is not None:
            return ' between {} and {}'.format(self.COVARIATE_BEFORE_DISPLAY, self.COVARIATE_AFTER_DISPLAY)
        else:
            s = ' between +${}^o\mathrm{C}$ and +${}^o\mathrm{C}$' if self.temporal_covariate_for_fit is AnomalyTemperatureWithSplineTemporalCovariate \
                else ' between {} and {}'
            s = s.format(self.covariate_before, self.covariate_after,
                         **d_temperature)
            return s

    def relative_changes_of_moment(self, altitudes, order=1, covariate_before=None, covariate_after=None):
        covariate_after, covariate_before = self.set_covariate_before_and_after(covariate_after, covariate_before)
        relative_changes = []
        for altitude in altitudes:
            mean_after = self.get_moment_covariate_float_or_list(altitude, covariate_after, order)
            mean_before = self.get_moment_covariate_float_or_list(altitude, covariate_before, order)
            relative_change = 100 * (mean_after - mean_before) / np.abs(mean_before)
            relative_changes.append(relative_change)
        return relative_changes

    # Minimizing the AIC and some properties

    @cached_property
    def sorted_estimators_with_default_selection_method(self):
        return self._sorted_estimators_with_method_name(method_name=self.SELECTION_METHOD_NAME)

    def method_name_to_best_estimator(self, method_names):
        return {self._sorted_estimators_with_method_name(method_name) for method_name in method_names}

    def _sorted_estimators_with_method_name(self, method_name) -> List[LinearMarginEstimator]:
        estimators = self.estimators_quality_checked
        if len(estimators) == 1:
            return estimators
        else:
            try:
                if OneFoldFit.SELECTION_METHOD_NAME == 'split_sample':
                    for estimator in estimators:
                        estimator.split_sample = compute_mean_log_score_with_split_sample(estimator)
                    assert not all([np.isinf(estimator.split_sample) for estimator in estimators])
                sorted_estimators = sorted([estimator for estimator in estimators],
                                           key=lambda e: e.__getattribute__(method_name))
            except AssertionError as e:
                print('Error for:\n', self.massif_name, self.altitude_group)
                raise e
            return sorted_estimators

    @cached_property
    def estimators_quality_checked(self):
        well_defined_estimators = []
        for estimator in self.fitted_estimators:
            if self.remove_physically_implausible_models:
                # Remove wrong shape
                if not (-0.5 < self._compute_shape_for_reference_altitude(estimator) < 0.5):
                    continue
                # Remove models with undefined parameters for the coordinate of interest
                coordinate_values_for_the_fit = estimator.coordinates_for_nllh
                if isinstance(self.altitude_group, DefaultAltitudeGroup):
                    coordinate_values_for_the_result = []
                else:
                    coordinate_values_for_the_result = [np.array([self.altitude_group.reference_altitude, c])
                                                        for c in self._covariate_before_and_after]
                coordinate_values_to_check = list(coordinate_values_for_the_fit) + coordinate_values_for_the_result
                has_undefined_parameters = False
                for coordinate in coordinate_values_to_check:
                    gev_params = estimator.margin_function_from_fit.get_params(coordinate)
                    if gev_params.has_undefined_parameters:
                        has_undefined_parameters = True
                        break
                if has_undefined_parameters:
                    continue
            #  Apply the goodness of fit
            if self.only_models_that_pass_goodness_of_fit_test:
                if not self.goodness_of_fit_test(estimator):
                    continue
            # Append to the list
            well_defined_estimators.append(estimator)
        # print('well defined estimators for {} m:'.format(self.altitude_group.reference_altitude),
        #       len(well_defined_estimators),'out of', len(self.fitted_estimators))
        # for e in well_defined_estimators:
        #     print(get_display_name_from_object_type(type(e.margin_model)))
        #     print(e.margin_model.param_name_to_climate_coordinates_with_effects)
        if len(well_defined_estimators) == 0:
            print(self.massif_name,
                  " has only implausible models for altitude={}".format(self.altitude_group.reference_altitude))
        # Check the number of models when we do not apply any goodness of fit
        if not (self.remove_physically_implausible_models or self.only_models_that_pass_goodness_of_fit_test):
            assert len(well_defined_estimators) == len(self.models_classes) * len(self.sub_combinations)
        return well_defined_estimators

    def get_coordinate(self, altitude, year):
        if isinstance(self.altitude_group, DefaultAltitudeGroup):
            if isinstance(altitude, tuple):
                coordinate = [year] + list(altitude)
            else:
                coordinate = np.array([year])
        else:
            coordinate = np.array([altitude, year])
        return coordinate

    def _compute_shape_for_reference_altitude(self, estimator):
        coordinate = self.get_coordinate(self.altitude_plot, self.covariate_after)
        gev_params = estimator.margin_function_from_fit.get_params(coordinate, is_transformed=False)
        shape = gev_params.shape
        return shape

    @property
    def has_at_least_one_valid_model(self):
        return len(self.sorted_estimators_with_default_selection_method) > 0

    @property
    def model_class_and_combination_to_estimator_with_finite_aic(self):
        return self._create_d(self.sorted_estimators_with_default_selection_method)

    @property
    def model_class_to_stationary_estimator_not_checked(self):
        return self._create_d(self.fitted_estimators)

    @staticmethod
    def _create_d(estimators):
        d = {}
        for estimator in estimators:
            margin_model = estimator.margin_model
            combination = load_combination(margin_model.param_name_to_climate_coordinates_with_effects)
            d[(type(margin_model), combination)] = estimator
        return d

    @property
    def best_estimator(self):
        if self.has_at_least_one_valid_model:
            best_estimator = self.sorted_estimators_with_default_selection_method[0]
            # Add some check up for the paper 2
            if not isinstance(self.altitude_group, DefaultAltitudeGroup):
                coordinate = best_estimator.coordinates_for_nllh[0]
                gev_params = best_estimator.margin_function_from_fit.get_params(coordinate)
                assert -0.5 < gev_params.shape < 0.5
                assert not self.only_models_that_pass_goodness_of_fit_test
                assert not self.remove_physically_implausible_models
            return best_estimator
        else:
            raise ValueError('This object should not have been called because '
                             'has_at_least_one_valid_model={}'.format(self.has_at_least_one_valid_model))

    @property
    def best_margin_model(self):
        return self.best_estimator.margin_model

    @property
    def best_margin_function_from_fit(self) -> IndependentMarginFunction:
        return self.best_estimator.margin_function_from_fit

    @property
    def best_shape(self):
        return self.get_gev_params(altitude=self.altitude_plot, year=self.last_year).shape

    @property
    def altitude_plot(self):
        return self.altitude_group.reference_altitude

    def best_coef(self, param_name, dim, degree):
        try:
            coef = self.best_margin_function_from_fit.param_name_to_coef[param_name]  # type: PolynomialAllCoef
            if coef.dim_to_polynomial_coef is None:
                return coef.intercept
            else:
                coef = coef.dim_to_polynomial_coef[dim]  # type: PolynomialCoef
                coef = coef.idx_to_coef[degree]
            return coef
        except (TypeError, KeyError):
            return None

    @property
    def model_names(self):
        return [e.margin_model.name_str for e in self.sorted_estimators_with_default_selection_method]

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
    def best_combination(self):
        return load_combination(
            self.best_estimator.margin_model.param_name_to_climate_coordinates_with_effects)

    @cached_property
    def stationary_estimator(self):
        if isinstance(self.altitude_group, DefaultAltitudeGroup):
            model_class = StationaryTemporalModel
        else:
            model_class = StationaryAltitudinal
        param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
            self.best_combination)
        return self.fitted_linear_margin_estimator(model_class, self.dataset,
                                                   param_name_to_climate_coordinates_with_effects)

    @cached_property
    def non_stationary_estimator_without_the_correction(self):
        combination = (0, 0, 0)
        model_class = type(self.best_estimator.margin_model)
        param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
            combination)
        return self.fitted_linear_margin_estimator(model_class, self.dataset,
                                                   param_name_to_climate_coordinates_with_effects)

    @cached_property
    def non_stationary_estimator_without_the_gcm_correction(self):
        assert self.param_name_to_climate_coordinates_with_effects is not None
        combination = self.best_combination
        model_class = type(self.best_estimator.margin_model)
        assert any([c in [1, 3] for c in combination])
        combination_without_gcm_correction = []
        for c in combination:
            if c == 1:
                res = 0
            elif c == 3:
                res = 2
            else:
                res = c
            combination_without_gcm_correction.append(res)
        param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
            combination_without_gcm_correction)
        return self.fitted_linear_margin_estimator(model_class, self.dataset,
                                                   param_name_to_climate_coordinates_with_effects)

    @cached_property
    def non_stationary_estimator_without_the_rcm_correction(self):
        assert self.param_name_to_climate_coordinates_with_effects is not None
        combination = self.best_combination
        model_class = type(self.best_estimator.margin_model)
        if any([c in [2, 3] for c in combination]):
            combination_without_rcm_correction = []
            for c in combination:
                if c == 2:
                    res = 0
                elif c == 3:
                    res = 1
                else:
                    res = c
                combination_without_rcm_correction.append(res)
            param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                combination_without_rcm_correction)
            return self.fitted_linear_margin_estimator(model_class, self.dataset,
                                                       param_name_to_climate_coordinates_with_effects)
        else:
            return self.best_estimator

    @property
    def correction_is_significant(self):
        assert self.param_name_to_climate_coordinates_with_effects is not None
        return self.likelihood_ratio_test(self.non_stationary_estimator_without_the_correction)

    @property
    def gcm_correction_is_significant(self):
        return self.likelihood_ratio_test(self.non_stationary_estimator_without_the_gcm_correction)

    @property
    def rcm_correction_is_significant(self):
        return self.likelihood_ratio_test(self.non_stationary_estimator_without_the_rcm_correction)

    def likelihood_ratio_test(self, estimator):
        degree_freedom_chi2 = self.best_estimator.nb_params - estimator.nb_params
        likelihood_ratio = estimator.deviance - self.best_estimator.deviance
        # pvalue = 1 - chi2.cdf(likelihood_ratio, df=degree_freedom_chi2)
        # print(likelihood_ratio, chi2.ppf(q=1 - self.SIGNIFICANCE_LEVEL, df=degree_freedom_chi2))
        # print("here likelihood ratio test: pvalue={}, significance level={}".format(pvalue, self.SIGNIFICANCE_LEVEL))
        return likelihood_ratio > chi2.ppf(q=1 - self.SIGNIFICANCE_LEVEL, df=degree_freedom_chi2)

    @cached_property
    def is_significant(self) -> bool:
        if isinstance(self.altitude_group, DefaultAltitudeGroup):
            # Likelihood ratio based significance
            print("likelihood ratio based significance")
            stationary_model_classes = [StationaryAltitudinal, StationaryGumbelAltitudinal,
                                        AltitudinalShapeLinearTimeStationary]
            if any([isinstance(self.best_estimator.margin_model, c)
                    for c in stationary_model_classes]):
                return False
            else:
                return self.likelihood_ratio_test(self.stationary_estimator)
        else:
            # Bootstrap based significance
            return self.cached_results_from_bootstrap[0]

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
        try:
            goodness_of_fit_anderson_test = goodness_of_fit_anderson(quantiles, self.SIGNIFICANCE_LEVEL)
        except SafeRunException as e:
            print('goodness of fit failed:', e.__repr__())
            goodness_of_fit_anderson_test = False
        return goodness_of_fit_anderson_test

    def standard_gumbel_quantiles(self, n=None):
        standard_gumbel_distribution = GevParams(loc=0, scale=1, shape=0)
        if n is None:
            n = len(self.dataset.coordinates)
        standard_gumbel_quantiles = [standard_gumbel_distribution.quantile(i / (n + 1)) for i in range(1, n + 1)]
        return standard_gumbel_quantiles

    def best_confidence_interval(self, altitude, year) -> EurocodeConfidenceIntervalFromExtremes:
        coordinate = self.get_coordinate(altitude, year)
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
        return self.best_estimator.sorted_empirical_standard_gumbel_quantiles()

    @cached_property
    def cached_results_from_bootstrap(self):
        start = time.time()
        bootstrap_fitted_functions = self.bootstrap_fitted_functions_from_fit
        end1 = time.time()
        duration = str(datetime.timedelta(seconds=end1 - start))
        print('Fit duration', duration)

        # First result - Compute the significance
        sign_of_changes = [self.sign_of_change(f) for f in bootstrap_fitted_functions]
        if self.sign_of_change(self.best_margin_function_from_fit) > 0:
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
                coordinate = self.get_coordinate(altitude, year)
                mean_estimate = self.get_return_level(self.best_margin_function_from_fit, coordinate)
                bootstrap_return_levels = [self.get_return_level(f, coordinate) for f in
                                           bootstrap_fitted_functions]
                confidence_interval = tuple([np.quantile(bootstrap_return_levels, q)
                                             for q in AbstractExtractEurocodeReturnLevel.bottom_and_upper_quantile])
                altitude_and_year_to_return_level_mean_estimate[key] = mean_estimate
                altitude_and_year_to_return_level_confidence_interval[key] = confidence_interval

        return is_significant, altitude_and_year_to_return_level_mean_estimate, altitude_and_year_to_return_level_confidence_interval

    @property
    def return_level_last_temporal_coordinate(self):
        df_temporal_covariate = self.dataset.coordinates.df_temporal_coordinates_for_fit(
            temporal_covariate_for_fit=self.temporal_covariate_for_fit,
            drop_duplicates=False)
        last_temporal_coordinate = df_temporal_covariate.loc[:, AbstractCoordinates.COORDINATE_T].max()
        # todo: améliorer the last temporal coordinate. on recupère la liste des rights_limits, puis on prend la valeur juste au dessus ou égale."""
        print('last temporal coordinate', last_temporal_coordinate)
        altitude = self.altitude_group.reference_altitude
        coordinate = self.get_coordinate(altitude, last_temporal_coordinate)
        return self.get_return_level(self.best_margin_function_from_fit, coordinate)

    @cached_property
    def bootstrap_fitted_functions_from_fit_cached(self):
        return self.bootstrap_fitted_functions_from_fit

    @property
    def bootstrap_fitted_functions_from_fit(self):
        print('nb of bootstrap for confidence interval=', AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP)
        idxs = list(range(AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP))

        if self.multiprocessing is None:
            start = time.time()
            with Pool(self.nb_cores_for_multiprocess) as p:
                batchsize = math.ceil(AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP / self.nb_cores_for_multiprocess)
                if self.max_batchsize is not None:
                    batchsize = min(self.max_batchsize, batchsize)
                list_functions_from_fit = p.map(self.fit_batch_bootstrap_estimator, batch(idxs, batchsize=batchsize))
                functions_from_fit = list(chain.from_iterable(list_functions_from_fit))

        elif self.multiprocessing:
            print('multiprocessing')
            start = time.time()
            with Pool(self.nb_cores_for_multiprocess) as p:
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

        observations = AnnualMaxima.from_df_coordinates(coordinate_values_to_maxima,
                                                        self.best_estimator.df_coordinates_for_fit)
        dataset = AbstractDataset(observations=observations, coordinates=self.dataset.coordinates)
        model_class = type(self.best_margin_model)

        function_from_fit = self.fitted_linear_margin_estimator(model_class, dataset,
                                                                self.param_name_to_climate_coordinates_with_effects).margin_function_from_fit

        return function_from_fit
