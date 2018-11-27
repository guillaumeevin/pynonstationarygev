from typing import Dict

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class LinearCoef(object):
    """
    Object that maps each dimension to its corresponding coefficient.
        dim = 0 correspond to the intercept
        dim = 1 correspond to the coordinate X
        dim = 2 correspond to the coordinate Y
        dim = 3 correspond to the coordinate Z
    """

    def __init__(self, gev_param_name: str, default_value: float = 0.0, dim_to_coef: Dict[int, float] = None):
        self.gev_param_name = gev_param_name
        self.dim_to_coef = dim_to_coef
        self.default_value = default_value

    def get_coef(self, dim: int) -> float:
        if self.dim_to_coef is None:
            return self.default_value
        else:
            return self.dim_to_coef.get(dim, self.default_value)

    @property
    def intercept(self):
        return self.get_coef(dim=0)

    @classmethod
    def from_dict(cls, coef_dict: Dict[int, float], gev_param_name: str, default_value: float = 0.0):
        pass

    def coef_dict(self, linear_dims):
        coef_dict = {}
        coef_template_str = self.gev_param_name + 'Coeff{}'
        # Constant param must be specified for all the parameters
        coef_dict[coef_template_str.format(1)] = self.intercept
        # Specify only the param that belongs to dim_to_coef
        for j, dim in enumerate(linear_dims, 2):
            coef_dict[coef_template_str.format(j)] = self.dim_to_coef[dim]
        return coef_dict

    def form_dict(self, linear_dims):
        """
        Example of formula that could be specified:
        loc.form = loc ~ coord_x
        scale.form = scale ~ coord_y
        shape.form = shape ~ coord_x+coord_y
        :return:
        """
        dim_to_name = {i: name for i, name in enumerate(AbstractCoordinates.COORDINATE_NAMES, 1)}
        formula_str = '1' if not linear_dims else '+'.join([dim_to_name[dim] for dim in linear_dims])
        return {self.gev_param_name + '.form': self.gev_param_name + ' ~ ' + formula_str}