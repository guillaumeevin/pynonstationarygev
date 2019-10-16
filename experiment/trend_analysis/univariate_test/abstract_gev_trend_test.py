import numpy as np
import pandas as pd
from cached_property import cached_property
from scipy.stats import chi2

from experiment.trend_analysis.univariate_test.abstract_univariate_test import AbstractUnivariateTest
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    StationaryStationModel
from extreme_fit.model.utils import SafeRunException
from extreme_fit.distribution.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class AbstractGevTrendTest(AbstractUnivariateTest):
    RRunTimeError_TREND = 'R RunTimeError trend'
    # I should use the quantile from the Eurocode for the buildings
    quantile_for_strength = 0.98
    nb_years_for_quantile_evolution = 10

    def __init__(self, years, maxima, starting_year, unconstrained_model_class, constrained_model_class=StationaryStationModel,
                 fit_method=AbstractTemporalLinearMarginModel.ISMEV_GEV_FIT_METHOD_STR):
        super().__init__(years, maxima, starting_year)
        self.fit_method = fit_method
        # Load observations, coordinates and datasets
        df = pd.DataFrame({AbstractCoordinates.COORDINATE_T: years})
        df_maxima_gev = pd.DataFrame(maxima, index=df.index)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df_maxima_gev)
        self.coordinates = AbstractTemporalCoordinates.from_df(df, transformation_class=CenteredScaledNormalization)
        self.dataset = AbstractDataset(observations=observations, coordinates=self.coordinates)

        try:
            # Fit constrained model
            constrained_model = constrained_model_class(self.coordinates, starting_point=self.starting_year, fit_method=self.fit_method)
            self.constrained_estimator = LinearMarginEstimator(self.dataset, constrained_model)
            self.constrained_estimator.fit()
            # Fit unconstrained model
            unconstrained_model = unconstrained_model_class(self.coordinates, starting_point=self.starting_year, fit_method=self.fit_method)
            self.unconstrained_estimator = LinearMarginEstimator(self.dataset, unconstrained_model)
            self.unconstrained_estimator.fit()
            self.crashed = False
        except SafeRunException:
            self.crashed = True

    # Type of trends

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
    def test_trend_type(self) -> str:
        if self.crashed:
            return self.RRunTimeError_TREND
        else:
            return super().test_trend_type

    # Likelihood ratio test

    @property
    def is_significant(self) -> bool:
        return self.likelihood_ratio > chi2.ppf(q=1 - self.SIGNIFICANCE_LEVEL, df=self.degree_freedom_chi2)

    @property
    def degree_freedom_chi2(self) -> int:
        raise NotImplementedError

    @property
    def likelihood_ratio(self):
        return self.unconstrained_model_deviance - self.constrained_model_deviance

    @property
    def constrained_model_deviance(self):
        if self.crashed:
            return np.nan
        else:
            return self.constrained_estimator.result_from_model_fit.deviance

    @property
    def unconstrained_model_deviance(self):
        if self.crashed:
            return np.nan
        else:
            return self.unconstrained_estimator.result_from_model_fit.deviance

    @property
    def unconstained_nllh(self):
        if self.crashed:
            return np.nan
        else:
            return self.unconstrained_estimator.result_from_model_fit.nllh

    # Evolution of the GEV parameters and corresponding quantiles

    @property
    def test_sign(self) -> int:
        return np.sign(self.test_trend_slope_strength)

    def get_non_stationary_linear_coef(self, gev_param_name: str):
        return self.unconstrained_estimator.margin_function_from_fit.get_coef(gev_param_name,
                                                                              AbstractCoordinates.COORDINATE_T)

    @cached_property
    def non_stationary_constant_gev_params(self) -> GevParams:
        # Constant parameters correspond to the gev params in 1958
        return self.unconstrained_estimator.margin_function_from_fit.get_gev_params(coordinate=np.array([1958]),
                                                                                    is_transformed=False)

    @cached_property
    def stationary_constant_gev_params(self) -> GevParams:
        # Constant parameters correspond to any gev params
        return self.constrained_estimator.margin_function_from_fit.get_gev_params(coordinate=np.array([1958]),
                                                                                  is_transformed=False)


    @property
    def test_trend_slope_strength(self):
        if self.crashed:
            return 0.0
        else:
            # Compute the slope strength
            slope = self._slope_strength()
            # Delta T must in the same unit as were the parameter of slope mu1 and sigma1
            slope *= self.nb_years_for_quantile_evolution * self.coordinates.transformed_distance_between_two_successive_years[0]
            return slope

    def _slope_strength(self):
        raise NotImplementedError

    @staticmethod
    def same_sign(a, b):
        return (a > 0 and b > 0) or (a < 0 and b < 0)

    @property
    def mean_difference_same_sign_as_slope_strenght(self) -> bool:
        return False

    @property
    def variance_difference_same_sign_as_slope_strenght(self) -> bool:
        return False

    def mean_difference(self, zeta0: float, mu1: float = 0.0, sigma1: float = 0.0) -> float:
        return GevParams(loc=mu1, scale=sigma1, shape=zeta0, accept_zero_scale_parameter=True).mean

    @property
    def test_trend_constant_quantile(self):
        if self.crashed:
            return 0.0
        else:
            return self.non_stationary_constant_gev_params.quantile(p=self.quantile_for_strength)



