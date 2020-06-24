from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import PolynomialMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractSpatioTemporalPolynomialModel(PolynomialMarginModel):

    def __init__(self, coordinates: AbstractCoordinates, params_user=None, starting_point=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle, nb_iterations_for_bayesian_fit=5000,
                 params_initial_fit_bayesian=None, type_for_MLE="GEV", params_class=GevParams, max_degree=2):
        super().__init__(coordinates, params_user, starting_point, fit_method, nb_iterations_for_bayesian_fit,
                         params_initial_fit_bayesian, type_for_MLE, params_class, max_degree)
        self.drop_duplicates = False


class NonStationaryLocationSpatioTemporalLinearityModel(AbstractSpatioTemporalPolynomialModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [
            (self.coordinates.idx_temporal_coordinates, 1),
            (self.coordinates.idx_x_coordinates, 1),
        ]})
