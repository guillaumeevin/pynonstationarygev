import numpy as np
import pandas as pd

from extreme_estimator.R_model.abstract_model import AbstractModel
from extreme_estimator.R_model.margin_model.gev_mle_fit import GevMleFit, mle_gev


class AbstractMarginModel(AbstractModel):
    GEV_SCALE = GevMleFit.GEV_SCALE
    GEV_LOCATION = GevMleFit.GEV_LOCATION
    GEV_SHAPE = GevMleFit.GEV_SHAPE
    GEV_PARAMETERS = [GEV_LOCATION, GEV_SCALE, GEV_SHAPE]

    def __init__(self, params_start_fit=None, params_sample=None):
        super().__init__(params_start_fit, params_sample)

    # Define the method to sample/fit a single gev

    def rgev(self, nb_obs, loc, scale, shape):
        gev_params = {
            self.GEV_LOCATION: loc,
            self.GEV_SCALE: scale,
            self.GEV_SHAPE: shape,
        }
        return self.r.rgev(nb_obs, **gev_params)

    def fitgev(self, x_gev, estimator=GevMleFit):
        mle_params = mle_gev(x_gev=x_gev)

    def gev_params_sample(self, coordinate) -> dict:
        pass

    # Define the method to sample/fit all marginals globally in the child classes

    def fitmargin(self, maxima, coord):
        df_fit_gev_params = None
        return df_fit_gev_params

    def rmargin(self, nb_obs, coord):
        maxima_gev = None
        return maxima_gev

    def frech2gev(self, maxima_frech: np.ndarray, coordinates: np.ndarray):
        assert len(maxima_frech) == len(coordinates)
        maxima_gev = []
        for x_frech, coordinate in zip(maxima_frech, coordinates):
            gev_params = self.gev_params_sample(coordinate)
            x_gev = self.r.frech2gev(x_frech, **gev_params)
            maxima_gev.append(x_gev)
        return np.array(maxima_gev)

    def gev2frech(self, maxima_gev: np.ndarray, df_gev_params: pd.DataFrame):
        assert len(maxima_gev) == len(df_gev_params)
        maxima_frech = []
        for x_gev, (_, s_gev_params) in zip(maxima_gev, df_gev_params.iterrows()):
            gev_params = dict(s_gev_params)
            gev2frech_param = {'emp': False}
            x_frech = self.r.gev2frech(x_gev, **gev_params, **gev2frech_param)
            maxima_frech.append(x_frech)
        return np.array(maxima_frech)


class SmoothMarginModel(AbstractMarginModel):
    pass


class ConstantMarginModel(SmoothMarginModel):
    def __init__(self, params_start_fit=None, params_sample=None):
        super().__init__(params_start_fit, params_sample)
        self.default_params_sample = {gev_param: 1.0 for gev_param in self.GEV_PARAMETERS}
        self.default_params_start_fit = {gev_param: 1.0 for gev_param in self.GEV_PARAMETERS}

    def gev_params_sample(self, coordinate):
        return self.default_params_sample

    def fitmargin(self, maxima, coord):
        return pd.DataFrame([pd.Series(self.default_params_start_fit) for _ in maxima])




