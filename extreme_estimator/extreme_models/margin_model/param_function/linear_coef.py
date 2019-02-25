from typing import Dict, List

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class LinearCoef(object):
    """
    Object that maps each dimension to its corresponding coefficient.
        dim = 0 correspond to the intercept
        dim = 1 correspond to the first coordinate
        dim = 2 correspond to the second coordinate
        dim = 3 correspond to the third coordinate...
    """
    INTERCEPT_NAME = 'intercept'
    COEFF_STR = 'Coeff'

    def __init__(self, gev_param_name: str, dim_to_coef: Dict[int, float] = None, default_value: float = 0.0):
        self.gev_param_name = gev_param_name
        self.dim_to_coef = dim_to_coef
        self.default_value = default_value

    def get_coef(self, dim: int) -> float:
        if self.dim_to_coef is None:
            return self.default_value
        else:
            return self.dim_to_coef.get(dim, self.default_value)

    @property
    def intercept(self) -> float:
        return self.get_coef(dim=0)

    @classmethod
    def coef_template_str(cls, gev_param_name: str, coefficient_name: str) -> str:
        """
        Example of coef that can be specified
            -for the spatial covariates: locCoeff
            -for the temporal covariates: tempCoeffLoc
        :param gev_param_name:
        :param coefficient_name:
        :return:
        """
        assert coefficient_name == cls.INTERCEPT_NAME or coefficient_name in AbstractCoordinates.COORDINATES_NAMES
        if coefficient_name == cls.INTERCEPT_NAME or coefficient_name in AbstractCoordinates.COORDINATE_SPATIAL_NAMES:
            coef_template_str = gev_param_name + cls.COEFF_STR + '{}'
        else:
            coef_template_str = 'temp' + cls.COEFF_STR + gev_param_name.title() + '{}'
        assert cls.COEFF_STR in coef_template_str
        return coef_template_str

    @staticmethod
    def has_dependence_in_spatial_coordinates(dim_to_coefficient_name):
        return any([coefficient_name in AbstractCoordinates.COORDINATE_SPATIAL_NAMES
                    for coefficient_name in dim_to_coefficient_name.values()])

    @classmethod
    def add_intercept_dim(cls, dims):
        return [0] + dims

    @classmethod
    def from_coef_dict(cls, coef_dict: Dict[str, float], gev_param_name: str, linear_dims: List[int],
                       dim_to_coefficient_name: Dict[int, str]):
        dims = cls.add_intercept_dim(linear_dims)
        dim_to_coef = {}
        j = 1
        for dim in dims:
            coefficient_name = dim_to_coefficient_name[dim]
            if coefficient_name == AbstractCoordinates.COORDINATE_T:
                j = 1
            coef = coef_dict[cls.coef_template_str(gev_param_name, coefficient_name).format(j)]
            dim_to_coef[dim] = coef
            j += 1
        return cls(gev_param_name, dim_to_coef)

    def coef_dict(self, linear_dims, dim_to_coefficient_name: Dict[int, str]) -> Dict[str, float]:
        dims = self.add_intercept_dim(linear_dims)
        coef_dict = {}
        j = 1
        for dim in dims:
            coefficient_name = dim_to_coefficient_name[dim]
            if coefficient_name == AbstractCoordinates.COORDINATE_T:
                j = 1
            coef = self.dim_to_coef[dim]
            coef_dict[self.coef_template_str(self.gev_param_name, coefficient_name).format(j)] = coef
            j += 1
        return coef_dict

    def form_dict(self, names: List[str]) -> Dict[str, str]:
        formula_str = '1' if not names else '+'.join(names)
        return {self.gev_param_name + '.form': self.gev_param_name + ' ~ ' + formula_str}

    def spatial_form_dict(self, coordinate_spatial_names: List[str]) -> Dict[str, str]:
        """
        Example of formula that could be specified:
        loc.form = loc ~ 1
        loc.form = loc ~ coord_x
        scale.form = scale ~ coord_y
        shape.form = shape ~ coord_x + coord_y
        :return:
        """
        assert all([name in AbstractCoordinates.COORDINATE_SPATIAL_NAMES for name in coordinate_spatial_names])
        return self.form_dict(coordinate_spatial_names)

    def temporal_form_dict(self, coordinate_temporal_names: List[str]) -> Dict[str, str]:
        """
        Example of formula that could be specified:
        temp.form.loc = loc ~ coord_t
        Example of formula that could not be specified
        temp.loc.form = shape ~ 1
        :return:
        """
        assert all([name in [AbstractCoordinates.COORDINATE_T] for name in coordinate_temporal_names])
        k, v = self.form_dict(coordinate_temporal_names).popitem()
        k = 'temp.form.' + k.split('.')[0]
        v = 'NULL' if '1' in v else v
        return {k: v}
