from extreme_fit.model.margin_model.polynomial_margin_model.altitudinal_models import StationaryAltitudinal, \
    NonStationaryAltitudinalLocationLinear, NonStationaryAltitudinalLocationQuadratic, \
    NonStationaryAltitudinalLocationLinearScaleLinear, NonStationaryAltitudinalLocationQuadraticScaleLinear

ALTITUDINAL_MODELS = [
    StationaryAltitudinal,
    NonStationaryAltitudinalLocationLinear,
    NonStationaryAltitudinalLocationQuadratic,
    NonStationaryAltitudinalLocationLinearScaleLinear,
    NonStationaryAltitudinalLocationQuadraticScaleLinear
][:]
