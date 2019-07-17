import numpy as np
import pandas as pd
from scipy.stats import chi2

from experiment.trend_analysis.univariate_test.abstract_univariate_test import AbstractUnivariateTest
from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef
from extreme_estimator.extreme_models.margin_model.temporal_linear_margin_model import StationaryStationModel, \
    NonStationaryLocationStationModel, NonStationaryScaleStationModel, NonStationaryShapeStationModel
from extreme_estimator.extreme_models.utils import SafeRunException
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class AbstractGevChangePointTest(AbstractUnivariateTest):
    RRunTimeError_TREND = 'R RunTimeError trend'
    # I should use the quantile from the Eurocode for the buildings
    quantile_for_strength = 0.98
    nb_years_for_quantile_evolution = 10

    def __init__(self, years, maxima, starting_year, non_stationary_model_class, gev_param_name):
        super().__init__(years, maxima, starting_year)
        self.gev_param_name = gev_param_name
        df = pd.DataFrame({AbstractCoordinates.COORDINATE_T: years})
        df_maxima_gev = pd.DataFrame(maxima, index=df.index)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df_maxima_gev)
        self.coordinates = AbstractTemporalCoordinates.from_df(df,
                                                               transformation_class=CenteredScaledNormalization)  # type: AbstractTemporalCoordinates
        self.dataset = AbstractDataset(observations=observations, coordinates=self.coordinates)
        # For the moment, we chose:
        # -the 0.99 quantile (even if for building maybe we should use the 1/50 so 0.98 quantile)
        # -to see the evolution between two successive years

        try:
            # Fit stationary model
            self.stationary_estimator = LinearMarginEstimator(self.dataset, StationaryStationModel(self.coordinates))
            self.stationary_estimator.fit()

            # Fit non stationary model
            non_stationary_model = non_stationary_model_class(self.coordinates, starting_point=self.starting_year)
            self.non_stationary_estimator = LinearMarginEstimator(self.dataset, non_stationary_model)
            self.non_stationary_estimator.fit()
            self.crashed = False
        except SafeRunException:
            self.crashed = True

    @classmethod
    def real_trend_types(cls):
        return super().real_trend_types() + [cls.RRunTimeError_TREND]

    @classmethod
    def get_real_trend_types(cls, display_trend_type):
        real_trend_types = super().get_real_trend_types(display_trend_type)
        if display_trend_type is cls.NON_SIGNIFICATIVE_TREND:
            real_trend_types.append(cls.RRunTimeError_TREND)
        return real_trend_types

    @property
    def likelihood_ratio(self):
        return self.non_stationary_deviance - self.stationary_deviance

    @property
    def non_stationary_constant_gev_params(self) -> GevParams:
        return self.non_stationary_estimator.result_from_fit.constant_gev_params

    @property
    def stationary_deviance(self):
        if self.crashed:
            return np.nan
        else:
            return self.stationary_estimator.result_from_fit.deviance

    @property
    def non_stationary_deviance(self):
        if self.crashed:
            return np.nan
        else:
            return self.non_stationary_estimator.result_from_fit.deviance

    @property
    def non_stationary_nllh(self):
        if self.crashed:
            return np.nan
        else:
            return self.non_stationary_estimator.result_from_fit.nllh

    @property
    def is_significant(self) -> bool:
        return self.likelihood_ratio > chi2.ppf(q=1 - self.SIGNIFICANCE_LEVEL, df=1)

    # Add a trend type that correspond to run that crashed

    # @classmethod
    # def trend_type_to_style(cls):
    #     trend_type_to_style = super().trend_type_to_style()
    #     trend_type_to_style[cls.RRunTimeError_TREND] = 'b:'
    #     return trend_type_to_style

    @property
    def test_trend_type(self) -> str:
        if self.crashed:
            return self.RRunTimeError_TREND
        else:
            return super().test_trend_type

    def get_coef(self, estimator, coef_name):
        return estimator.margin_function_fitted.get_coef(self.gev_param_name, coef_name)

    @property
    def non_stationary_intercept_coef(self):
        return self.get_coef(self.non_stationary_estimator, LinearCoef.INTERCEPT_NAME)

    @property
    def non_stationary_linear_coef(self):
        return self.get_coef(self.non_stationary_estimator, AbstractCoordinates.COORDINATE_T)

    @property
    def test_trend_slope_strength(self):
        if self.crashed:
            return 0.0
        else:
            slope = self._slope_strength()
            slope *= self.nb_years_for_quantile_evolution * self.coordinates.transformed_distance_between_two_successive_years[0]
            return slope

    def _slope_strength(self):
        raise NotImplementedError

    @property
    def test_trend_constant_quantile(self):
        if self.crashed:
            return 0.0
        else:
            return self.non_stationary_constant_gev_params.quantile(p=self.quantile_for_strength)

    @property
    def percentage_of_change_per_year(self):
        ratio = np.abs(self.non_stationary_linear_coef) / np.abs(self.non_stationary_intercept_coef)
        scaled_ratio = ratio * self.coordinates.transformed_distance_between_two_successive_years
        percentage_of_change_per_year = 100 * scaled_ratio
        return percentage_of_change_per_year[0]

    @property
    def test_sign(self) -> int:
        return np.sign(self.non_stationary_linear_coef)


class GevLocationChangePointTest(AbstractGevChangePointTest):

    def __init__(self, years, maxima, starting_year):
        super().__init__(years, maxima, starting_year,
                         NonStationaryLocationStationModel, GevParams.LOC)

    def _slope_strength(self):
        return self.non_stationary_constant_gev_params.quantile_strength_evolution_ratio(p=self.quantile_for_strength,
                                                                                         mu1=self.non_stationary_linear_coef)


class GevScaleChangePointTest(AbstractGevChangePointTest):

    def __init__(self, years, maxima, starting_year):
        super().__init__(years, maxima, starting_year,
                         NonStationaryScaleStationModel, GevParams.SCALE)

    def _slope_strength(self):
        return self.non_stationary_constant_gev_params.quantile_strength_evolution_ratio(
            p=self.quantile_for_strength,
            sigma1=self.non_stationary_linear_coef)


class GevShapeChangePointTest(AbstractGevChangePointTest):

    def __init__(self, years, maxima, starting_year):
        super().__init__(years, maxima, starting_year,
                         NonStationaryShapeStationModel, GevParams.SHAPE)
