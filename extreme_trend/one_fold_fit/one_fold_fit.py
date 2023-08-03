import numpy as np

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.distribution.gumbel.gumbel_gof import goodness_of_fit_anderson, get_pvalue_anderson_darling_test
from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator_short
from extreme_fit.function.margin_function.independent_margin_function import IndependentMarginFunction
from extreme_fit.model.margin_model.spline_margin_model.spline_margin_model import SplineMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.utils import SafeRunException
from extreme_trend.one_fold_fit.altitude_group import DefaultAltitudeGroup
from projected_extremes.section_results.utils.combination_utils import load_combination, generate_sub_combination, \
    load_param_name_to_climate_coordinates_with_effects
from root_utils import NB_CORES
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class OneFoldFit(object):
    SIGNIFICANCE_LEVEL = 0.05
    return_period = 100
    quantile_level = 1 - (1 / return_period)
    multiprocessing = None
    nb_cores_for_multiprocess = NB_CORES
    max_batchsize = None
    COVARIATE_BEFORE_TEMPERATURE = 1
    COVARIATE_AFTER_TEMPERATURE = 2

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
        assert len(models_classes) == 1
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
        return estimator

    @classmethod
    def get_moment_str(cls, order):
        if order == 1:
            return 'mean annual maxima'
        elif order == 2:
            return 'std'
        elif order is None:
            return '{}-year return levels'.format(cls.return_period)

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
        gev_params = self.best_margin_function_from_fit.get_params(coordinate)
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

    def relative_changes_of_moment(self, altitudes, order=1, covariate_before=None, covariate_after=None):
        covariate_after, covariate_before = self.set_covariate_before_and_after(covariate_after, covariate_before)
        relative_changes = []
        for altitude in altitudes:
            mean_after = self.get_moment_covariate_float_or_list(altitude, covariate_after, order)
            mean_before = self.get_moment_covariate_float_or_list(altitude, covariate_before, order)
            relative_change = 100 * (mean_after - mean_before) / np.abs(mean_before)
            relative_changes.append(relative_change)
        return relative_changes

    @property
    def fitted_estimator(self):
        assert len(self.fitted_estimators) == 1
        return list(self.fitted_estimators)[0]

    def get_coordinate(self, altitude, year):
        if isinstance(self.altitude_group, DefaultAltitudeGroup):
            if isinstance(altitude, tuple):
                coordinate = [year] + list(altitude)
            else:
                coordinate = np.array([year])
        else:
            coordinate = np.array([altitude, year])
        return coordinate

    @property
    def best_estimator(self):
        return self.fitted_estimator

    @property
    def best_margin_function_from_fit(self) -> IndependentMarginFunction:
        return self.best_estimator.margin_function_from_fit

    @property
    def altitude_plot(self):
        return self.altitude_group.reference_altitude

    def goodness_of_fit_test_separated_for_each_gcm_rcm_couple(self, estimator):
        df = estimator.dataset.coordinates.df_coordinate_climate_model.loc[:, [AbstractCoordinates.COORDINATE_GCM,
                                                                                AbstractCoordinates.COORDINATE_RCM]]
        df = df.drop_duplicates()
        test_results = []
        test_names = []
        pvalues = []
        # Save the df of reference
        # df_all_coordinates_estimator = estimator.dataset.coordinates.df_all_coordinates.copy()
        df_coordinate_climate_model_estimator = estimator.dataset.coordinates.df_coordinate_climate_model.copy()
        df_maxima_gev_estimator = estimator.dataset.observations.df_maxima_gev.copy()
        df_coordinates_for_fit_estimator = estimator.df_coordinates_for_fit.copy()
        for j, (i, row) in enumerate(df.iterrows(), 1):
            # Load gcm and rcm of interest
            gcm, rcm = row[AbstractCoordinates.COORDINATE_GCM], row[AbstractCoordinates.COORDINATE_RCM]
            # print('loop ', j, ' for {} and {}'.format(gcm, rcm))
            test_names.append(str(gcm) + '_' + str(rcm))
            # Find the index for this row
            if isinstance(gcm, str):
                ind = df_coordinate_climate_model_estimator[AbstractCoordinates.COORDINATE_GCM] == gcm
                ind &= df_coordinate_climate_model_estimator[AbstractCoordinates.COORDINATE_RCM] == rcm
            else:
                ind = df_coordinate_climate_model_estimator[AbstractCoordinates.COORDINATE_GCM].isnull()
                ind &= df_coordinate_climate_model_estimator[AbstractCoordinates.COORDINATE_RCM].isnull()
            assert 0 < sum(ind) <= 150
            # Create an estimator with only the information on this row
            # estimator.dataset.coordinates.df_all_coordinates = df_all_coordinates_estimator.loc[ind].copy()
            estimator.dataset.coordinates.df_coordinate_climate_model = df_coordinate_climate_model_estimator.loc[ind].copy()
            estimator.dataset.observations.df_maxima_gev = df_maxima_gev_estimator.loc[ind].copy()
            coordinate_values = df_coordinates_for_fit_estimator.loc[ind].copy().values
            assert len(coordinate_values) <= 150, len(coordinate_values)
            # Run test with some specific coordinate for fit
            test_results.append(self.goodness_of_fit_test(estimator, coordinate_values))
            # Compute pvalue
            pvalues.append(get_pvalue_anderson_darling_test(estimator.sorted_empirical_standard_gumbel_quantiles(coordinate_values=coordinate_values)))
        estimator.dataset.observations.df_maxima_gev = df_maxima_gev_estimator
        estimator.dataset.coordinates.df_coordinate_climate_model = df_coordinate_climate_model_estimator
        return test_results, test_names, pvalues

    def goodness_of_fit_test(self, estimator, coordinate_values=None):
        if estimator.dataset.coordinates.has_several_climate_coordinates:
            test_results, *_ = self.goodness_of_fit_test_separated_for_each_gcm_rcm_couple(estimator)
            return all(test_results)
        else:
            quantiles = estimator.sorted_empirical_standard_gumbel_quantiles(coordinate_values=coordinate_values)
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