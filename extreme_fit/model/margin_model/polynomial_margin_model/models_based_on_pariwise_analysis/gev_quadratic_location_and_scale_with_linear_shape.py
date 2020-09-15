from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import AbstractAltitudinalModel


class AltitudinalShapeLinearTimeLocationQuadratic(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)],
            GevParams.SHAPE: [(self.coordinates.idx_x_coordinates, 1)]
        }


class AltitudinalShapeLinearTimeLocationQuadraticScaleLinear(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SHAPE: [(self.coordinates.idx_x_coordinates, 1)]
        }


class AltitudinalShapeLinearTimeLocationQuadraticScaleQuadratic(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SHAPE: [(self.coordinates.idx_x_coordinates, 1)]
        }


class AltitudinalShapeLinearTimeLocationLinearScaleQuadratic(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SHAPE: [(self.coordinates.idx_x_coordinates, 1)]
        }


class AltitudinalShapeLinearTimeScaleQuadratic(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SHAPE: [(self.coordinates.idx_x_coordinates, 1)]
        }
