from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import AbstractAltitudinalModel, \
    AbstractAddCrossTermForLocation
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import PolynomialMarginModel
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel


class AltitudinalOnlyScale(AbstractAltitudinalModel):
    pass

class StationaryAltitudinalOnlyScale(AltitudinalOnlyScale):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)],
        }




class NonStationaryAltitudinalOnlyScaleLocationLinear(AltitudinalOnlyScale):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        }


class NonStationaryAltitudinalOnlyScaleLocationQuadratic(AltitudinalOnlyScale):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        }

class NonStationaryAltitudinalOnlyScaleLocationCubic(AltitudinalOnlyScale):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 3)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        }

class NonStationaryAltitudinalOnlyScaleLocationOrder4(AltitudinalOnlyScale):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 4)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        }

# Add cross terms



class NonStationaryOnlyScaleCrossTermForLocation(AbstractAddCrossTermForLocation, StationaryAltitudinalOnlyScale):
    pass


class NonStationaryAltitudinalOnlyScaleLocationLinearCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                 NonStationaryAltitudinalOnlyScaleLocationLinear):
    pass


class NonStationaryAltitudinalOnlyScaleLocationQuadraticCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                    NonStationaryAltitudinalOnlyScaleLocationQuadratic):
    pass

class NonStationaryAltitudinalOnlyScaleLocationCubicCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                NonStationaryAltitudinalOnlyScaleLocationCubic,
                                                                ):
    pass

class NonStationaryAltitudinalOnlyScaleLocationOrder4CrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                NonStationaryAltitudinalOnlyScaleLocationOrder4,
                                                                ):
    pass

