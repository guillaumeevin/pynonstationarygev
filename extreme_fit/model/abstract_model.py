class AbstractModel(object):

    def __init__(self, use_start_value=False, params_start_fit=None, params_sample=None):
        self.default_params = None
        self.use_start_value = use_start_value
        self.user_params_start_fit = params_start_fit
        self.user_params_sample = params_sample

    @property
    def params_start_fit(self) -> dict:
        return self.default_params.copy()

    @property
    def params_sample(self) -> dict:
        return self.merge_params(default_params=self.default_params, input_params=self.user_params_sample)

    @staticmethod
    def merge_params(default_params, input_params):
        assert default_params is not None, 'some default_params need to be specified'
        merged_params = default_params.copy()
        if input_params is not None:
            assert isinstance(default_params, dict) and isinstance(input_params, dict)
            assert set(input_params.keys()).issubset(set(default_params.keys()))
            merged_params.update(input_params)
        return merged_params
