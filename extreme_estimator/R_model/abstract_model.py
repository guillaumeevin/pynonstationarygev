from extreme_estimator.R_model.utils import get_loaded_r


class AbstractModel(object):

    r = get_loaded_r()

    def __init__(self, params_start_fit=None, params_sample=None):
        self.default_params_start_fit = None
        self.default_params_sample = None
        self.user_params_start_fit = params_start_fit
        self.user_params_sample = params_sample