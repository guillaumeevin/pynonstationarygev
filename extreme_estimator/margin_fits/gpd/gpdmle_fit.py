import numpy as np

from extreme_estimator.margin_fits.gpd.gpd_params import GpdParams
from extreme_estimator.margin_fits.margin_fits_utils import spatial_extreme_gpdmle_fit


class GpdMleFit(object):

    def __init__(self, x_gev: np.ndarray, threshold):
        assert np.ndim(x_gev) == 1
        assert len(x_gev) > 1
        self.x_gev = x_gev
        self.threshold = threshold
        self.mle_params = spatial_extreme_gpdmle_fit(x_gev, threshold)
        self.gev_params = GpdParams.from_dict({**self.mle_params, 'threshold': threshold})