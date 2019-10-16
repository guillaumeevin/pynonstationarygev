from rpy2 import robjects

from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import \
    AbstractResultFromModelFit


class ResultFromExtremes(AbstractResultFromModelFit):

    def __init__(self, result_from_fit: robjects.ListVector, gev_param_name_to_dim=None) -> None:
        super().__init__(result_from_fit)
        self.gev_param_name_to_dim = gev_param_name_to_dim
        print(list(self.name_to_value.keys()))

    # @property
    # def

    # @property
    # def margin_coef_dict(self):
    #     assert self.gev_param_name_to_dim is not None
    #     # Build the Coeff dict from gev_param_name_to_dim
    #     coef_dict = {}
    #     i = 0
    #     mle_values = self.name_to_value['mle']
    #     for gev_param_name in GevParams.PARAM_NAMES:
    #         # Add intercept
    #         intercept_coef_name = LinearCoef.coef_template_str(gev_param_name, LinearCoef.INTERCEPT_NAME).format(1)
    #         coef_dict[intercept_coef_name] = mle_values[i]
    #         i += 1
    #         # Add a potential linear temporal trend
    #         if gev_param_name in self.gev_param_name_to_dim:
    #             temporal_coef_name = LinearCoef.coef_template_str(gev_param_name,
    #                                                               AbstractCoordinates.COORDINATE_T).format(1)
    #             coef_dict[temporal_coef_name] = mle_values[i]
    #             i += 1
    #     return coef_dict

    # @property
    # def all_parameters(self):
    #     return self.margin_coef_dict
    #
    # @property
    # def nllh(self):
    #     return convertFloatVector_to_float(self.name_to_value['nllh'])
    #
    # @property
    # def deviance(self):
    #     return - 2 * self.nllh
    #
    # @property
    # def convergence(self) -> str:
    #     return convertFloatVector_to_float(self.name_to_value['conv']) == 0

