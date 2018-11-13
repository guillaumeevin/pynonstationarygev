from extreme_estimator.R_fit.gev_fit.gev_mle_fit import GevMleFit, mle_gev
from extreme_estimator.R_fit.utils import get_loaded_r


def frechet_unitary_transformation(data, location, scale, shape):
    """
    Compute the unitary Frechet transformed data
    (1 + \zeta \frac{z - \mu}{\sigma}) ^ {\frac{1}{\zeta}}
    """
    assert False
    # todo: there is already a function doing that in R
    return (1 + shape * (data - location) / scale) ** (1 / shape)


class GevParameters(object):

    def __init__(self, location, scale, shape):
        self.location = location
        self.scale = scale
        self.shape = shape


def frechet_unitary_transformation_from_gev_parameters(data, gev_parameters: GevParameters):
    return frechet_unitary_transformation(data, gev_parameters.location)


class AbstractMarginModel(object):
    GEV_SCALE = GevMleFit.GEV_SCALE
    GEV_LOCATION = GevMleFit.GEV_LOCATION
    GEV_SHAPE = GevMleFit.GEV_SHAPE
    GEV_PARAMETERS = [GEV_LOCATION, GEV_SCALE, GEV_SHAPE]

    def __init__(self):
        """
        Class to fit a GEV a list of data. Compute also the corresponding unitary data

        :param coordinate: Represents the spatio-temporal spatial_coordinates of the marginals
        :param data:  array of data corresponding to this position (and potentially its neighborhood)
        """
        self.default_params = {gev_param: 1.0 for gev_param in self.GEV_PARAMETERS}
        self.r = get_loaded_r()

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

    # Define the method to sample/fit all marginals globally in the child classes

    def fitmargin(self, maxima, coord):
        df_gev_params = None
        return df_gev_params

    def rmargin(self, nb_obs, coord):
        pass

    def get_maxima(self, maxima_normalized, coord):
        pass

    def get_maxima_normalized(self, maxima, df_gev_params):
        pass


class SmoothMarginModel(AbstractMarginModel):
    pass


class ConstantMarginModel(SmoothMarginModel):
    pass
