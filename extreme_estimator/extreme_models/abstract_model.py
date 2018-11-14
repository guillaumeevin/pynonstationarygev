from extreme_estimator.extreme_models.utils import get_loaded_r


class AbstractModel(object):
    r = get_loaded_r()

    def __init__(self, params_start_fit=None, params_sample=None):
        self.default_params_start_fit = None
        self.default_params_sample = None
        self.user_params_start_fit = params_start_fit
        self.user_params_sample = params_sample

    @property
    def params_start_fit(self) -> dict:
        return self.merge_params(default_params=self.default_params_start_fit, input_params=self.user_params_start_fit)

    @property
    def params_sample(self) -> dict:
        return self.merge_params(default_params=self.default_params_sample, input_params=self.user_params_sample)

    @staticmethod
    def merge_params(default_params, input_params):
        assert default_params is not None, 'some default_params need to be specified'
        merged_params = default_params.copy()
        if input_params is not None:
            assert isinstance(default_params, dict) and isinstance(input_params, dict)
            assert set(input_params.keys()).issubset(set(default_params.keys()))
            merged_params.update(input_params)
        return merged_params
