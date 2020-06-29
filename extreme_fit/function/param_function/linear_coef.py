from typing import Dict, List

from extreme_fit.function.param_function.abstract_coef import AbstractCoef
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class LinearCoef(AbstractCoef):
    """
    Object that maps each dimension to its corresponding coefficient.
        dim = 0 correspond to the intercept
        dim = 1 correspond to the first coordinate
        dim = 2 correspond to the second coordinate
        dim = 3 correspond to the third coordinate...
    """
    INTERCEPT_NAME = 'intercept'
    COEFF_STR = 'Coeff'

    @property
    def intercept(self) -> float:
        return self.get_coef(idx=-1)

    @classmethod
    def coef_template_str(cls, param_name: str, coefficient_name: str) -> str:
        """
        Example of coef that can be specified
            -for the spatial covariates: locCoeff
            -for the temporal covariates: tempCoeffLoc
        :param param_name:
        :param coefficient_name:
        :return:
        """
        assert coefficient_name == cls.INTERCEPT_NAME or coefficient_name in AbstractCoordinates.COORDINATES_NAMES \
               or any([coordinate_name in coefficient_name for coordinate_name in AbstractCoordinates.COORDINATES_NAMES])
        if coefficient_name == cls.INTERCEPT_NAME or coefficient_name in AbstractCoordinates.COORDINATE_SPATIAL_NAMES:
            coef_template_str = param_name + cls.COEFF_STR + '{}'
        elif coefficient_name == AbstractCoordinates.COORDINATE_T:
            coef_template_str = 'temp' + cls.COEFF_STR + param_name.title() + '{}'
        elif len([c for c in AbstractCoordinates.COORDINATES_NAMES if c in coefficient_name]) >= 2:
            coef_template_str = 'cross' + cls.COEFF_STR + param_name.title() + '{}'
        else:
            raise NotImplementedError
        assert cls.COEFF_STR in coef_template_str
        return coef_template_str

    @classmethod
    def coefficient_name(cls, dim, dim_to_coordinate_name):
        if isinstance(dim, int):
            return dim_to_coordinate_name[dim]
        elif isinstance(dim, tuple):
            return ' * '.join([dim_to_coordinate_name[d] for d in dim])
        else:
            raise NotImplementedError

    @classmethod
    def offset_from_coefficient_name(cls, coefficient_name):
        return 1 if coefficient_name == AbstractCoordinates.COORDINATE_X else 0

    @staticmethod
    def has_dependence_in_spatial_coordinates(dim_to_coefficient_name):
        return any([coefficient_name in AbstractCoordinates.COORDINATE_SPATIAL_NAMES
                    for coefficient_name in dim_to_coefficient_name.values()])

    @classmethod
    def add_intercept_idx(cls, dims):
        return [-1] + dims

    """ Coef dict """

    @classmethod
    def from_coef_dict(cls, coef_dict: Dict[str, float], param_name: str, dims: List[int],
                       coordinates: AbstractCoordinates):
        idx_to_coef = {-1: coef_dict[cls.coef_template_str(param_name, coefficient_name=cls.INTERCEPT_NAME).format(1)]}
        j = 2
        for dim in dims:
            coefficient_name = coordinates.coordinates_names[dim]
            if coefficient_name == AbstractCoordinates.COORDINATE_T:
                j = 1
            coef = coef_dict[cls.coef_template_str(param_name, coefficient_name).format(j)]
            idx_to_coef[dim] = coef
            j += 1
        return cls(param_name=param_name, idx_to_coef=idx_to_coef)

    def coef_dict(self, dims, dim_to_coefficient_name: Dict[int, str]) -> Dict[str, float]:
        dims = self.add_intercept_idx(dims)
        coef_dict = {}
        j = 1
        for dim in dims:
            coefficient_name = dim_to_coefficient_name[dim]
            if coefficient_name == AbstractCoordinates.COORDINATE_T:
                j = 1
            coef = self.idx_to_coef[dim]
            coef_dict[self.coef_template_str(self.param_name, coefficient_name).format(j)] = coef
            j += 1
        return coef_dict

    def spatial_form_dict(self, coordinate_spatial_names: List[str], spatial_dims) -> Dict[str, str]:
        """
        Example of formula that could be specified:
        loc.form = loc ~ 1
        loc.form = loc ~ coord_x
        scale.form = scale ~ coord_y
        shape.form = shape ~ coord_x + coord_y
        :return:
        """
        assert all([name in AbstractCoordinates.COORDINATE_SPATIAL_NAMES for name in coordinate_spatial_names])
        return self.form_dict(coordinate_spatial_names, spatial_dims)

    def temporal_form_dict(self, coordinate_temporal_names: List[str], temporal_dims) -> Dict[str, str]:
        """
        Example of formula that could be specified:
        temp.form.loc = loc ~ coord_t
        Example of formula that could not be specified
        temp.loc.form = loc ~ 1
        :return:
        """
        assert all([name in [AbstractCoordinates.COORDINATE_T] for name in coordinate_temporal_names])
        k, v = self.form_dict(coordinate_temporal_names, temporal_dims).popitem()
        k = 'temp.form.' + k.split('.')[0]
        v = 'NULL' if '~ 1' in v else v
        return {k: v}
