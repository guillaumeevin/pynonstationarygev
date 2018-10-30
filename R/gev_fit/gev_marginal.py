from R.gev_fit.gev_mle_fit import GevMleFit


def frechet_unitary_transformation(data, location, scale, shape):
    """
    Compute the unitary Frechet transformed data
    (1 + \zeta \frac{z - \mu}{\sigma}) ^ {\frac{1}{\zeta}}
    """
    return (1 + shape * (data - location) / scale) ** (1 / shape)


class GevMarginal(object):

    def __init__(self, position, data, estimator=GevMleFit):
        """
        Class to fit a GEV a list of data. Compute also the corresponding unitary data

        :param position: Represents the spatio-temporal position of the marginals
        :param data:  array of data corresponding to this position (and potentially its neighborhood)
        """
        self.position = position
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
