class AbstractModel(object):

    def __init__(self, params_user=None):
        self.params_user = params_user

    @property
    def default_params(self):
        return None

    @property
    def params_sample(self) -> dict:
        return self.merge_params(default_params=self.default_params, params_user=self.params_user)

    @staticmethod
    def merge_params(default_params, params_user):
        assert default_params is not None, 'some default_params need to be specified'
        merged_params = default_params.copy()
        if params_user is not None:
            assert isinstance(default_params, dict) and isinstance(params_user, dict)
            assert set(params_user.keys()).issubset(set(default_params.keys()))
            merged_params.update(params_user)
        return merged_params
