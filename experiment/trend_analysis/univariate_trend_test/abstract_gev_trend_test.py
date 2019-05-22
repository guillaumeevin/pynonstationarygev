import numpy as np
import pandas as pd
from scipy.stats import chi2

from experiment.trend_analysis.univariate_trend_test.abstract_trend_test import AbstractTrendTest
from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
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


class AbstractGevTrendTest(AbstractTrendTest):
    RRunTimeError_TREND = 'R RunTimeError trend'

    def __init__(self, years_after_change_point, maxima_after_change_point, non_stationary_model_class):
        super().__init__(years_after_change_point, maxima_after_change_point)
        df = pd.DataFrame({AbstractCoordinates.COORDINATE_T: years_after_change_point})
        df_maxima_gev = pd.DataFrame(maxima_after_change_point, index=df.index)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df_maxima_gev)
        self.coordinates = AbstractTemporalCoordinates.from_df(df, transformation_class=CenteredScaledNormalization)
        self.dataset = AbstractDataset(observations=observations, coordinates=self.coordinates)

        try:
            # Fit stationary model
            self.stationary_estimator = LinearMarginEstimator(self.dataset, StationaryStationModel(self.coordinates))
            self.stationary_estimator.fit()

            # Fit non stationary model
            self.non_stationary_estimator = LinearMarginEstimator(self.dataset,
                                                                  non_stationary_model_class(self.coordinates))
            self.non_stationary_estimator.fit()
            self.crashed = False
        except SafeRunException:
            self.crashed = True

    @property
    def likelihood_ratio(self):
        return 2 * (self.non_stationary_estimator.result_from_fit.deviance -
                    self.stationary_estimator.result_from_fit.deviance)

    @property
    def is_significant(self) -> bool:
        return self.likelihood_ratio > chi2.ppf(q=1 - self.SIGNIFICANCE_LEVEL, df=1)

    # Add a trend type that correspond to run that crashed

    @classmethod
    def trend_type_to_style(cls):
        trend_type_to_style = super().trend_type_to_style()
        trend_type_to_style[cls.RRunTimeError_TREND] = 'b:'
        return trend_type_to_style

    @property
    def test_trend_type(self) -> str:
        if self.crashed:
            return self.RRunTimeError_TREND
        else:
            return super().test_trend_type


class GevLocationTrendTest(AbstractGevTrendTest):

    def __init__(self, years_after_change_point, maxima_after_change_point):
        super().__init__(years_after_change_point, maxima_after_change_point, NonStationaryLocationStationModel)

    @property
    def test_sign(self) -> int:
        return np.sign(self.non_stationary_estimator.margin_function_fitted.get_coef(GevParams.LOC,
                                                                                     AbstractCoordinates.COORDINATE_T))


class GevScaleTrendTest(AbstractGevTrendTest):

    def __init__(self, years_after_change_point, maxima_after_change_point):
        super().__init__(years_after_change_point, maxima_after_change_point, NonStationaryScaleStationModel)

    @property
    def test_sign(self) -> int:
        return np.sign(self.non_stationary_estimator.margin_function_fitted.get_coef(GevParams.SCALE,
                                                                                     AbstractCoordinates.COORDINATE_T))


class GevShapeTrendTest(AbstractGevTrendTest):

    def __init__(self, years_after_change_point, maxima_after_change_point):
        super().__init__(years_after_change_point, maxima_after_change_point, NonStationaryShapeStationModel)

    @property
    def test_sign(self) -> int:
        return np.sign(self.non_stationary_estimator.margin_function_fitted.get_coef(GevParams.SHAPE,
                                                                                     AbstractCoordinates.COORDINATE_T))
