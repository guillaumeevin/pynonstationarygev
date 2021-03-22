from extreme_fit.model.margin_model.linear_margin_model.margin_model_with_effect.abstract_margin_model_with_effect import \
    AbstractMarginModelWithEffect
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal


class MarginModelWithGcmEffect(AbstractMarginModelWithEffect):
    pass


class StationaryAltitudinalWithGCMEffect(MarginModelWithGcmEffect, StationaryAltitudinal):
    pass
