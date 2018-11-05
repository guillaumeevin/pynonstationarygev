from R.gev_fit.gev_mle_fit import GevMleFit


def frechet_unitary_transformation(data, location, scale, shape):
    """
    Compute the unitary Frechet transformed data
    (1 + \zeta \frac{z - \mu}{\sigma}) ^ {\frac{1}{\zeta}}
    """
    # todo: there is already a function doing that in R
    return (1 + shape * (data - location) / scale) ** (1 / shape)


class GevParameters(object):

    def __init__(self, location, scale, shape):
        self.location = location
        self.scale = scale
        self.shape = shape


def frechet_unitary_transformation_from_gev_parameters(data, gev_parameters: GevParameters):
    return frechet_unitary_transformation(data, gev_parameters.location)


class GevMarginal(object):

    def __init__(self, coordinate, data, estimator=GevMleFit):
        """
        Class to fit a GEV a list of data. Compute also the corresponding unitary data

        :param coordinate: Represents the spatio-temporal spatial_coordinates of the marginals
        :param data:  array of data corresponding to this position (and potentially its neighborhood)
        """
        self.coordinate = coordinate
        self.data = data
        self.gev_estimator = estimator(x_gev=data)
        self.gev_parameters_estimated = [self.location, self.scale, self.shape]
        self.frechet_unitary_data = frechet_unitary_transformation(data=data, location=self.location,
                                                                   scale=self.scale, shape=self.shape)

    @property
    def location(self):
        return self.gev_estimator.location

    @property
    def scale(self):
        return self.gev_estimator.scale

    @property
    def shape(self):
        return self.gev_estimator.shape
