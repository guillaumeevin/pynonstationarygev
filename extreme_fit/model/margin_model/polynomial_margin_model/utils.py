from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal, \
    NonStationaryAltitudinalLocationLinear, NonStationaryAltitudinalLocationQuadratic, \
    NonStationaryAltitudinalLocationLinearScaleLinear, NonStationaryAltitudinalLocationQuadraticScaleLinear, \
    NonStationaryCrossTermForLocation, NonStationaryAltitudinalLocationLinearCrossTermForLocation, \
    NonStationaryAltitudinalLocationQuadraticCrossTermForLocation, \
    NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForLocation, \
    NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation, NonStationaryAltitudinalScaleLinear, \
    NonStationaryAltitudinalScaleLinearCrossTermForLocation
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models_cross_term_in_scale import \
    NonStationaryAltitudinalLocationLinearScaleQuadraticCrossTermForScale, \
    NonStationaryAltitudinalLocationQuadraticScaleQuadraticCrossTermForScale, \
    NonStationaryAltitudinalScaleLinearCrossTermForScale, \
    NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForScale, \
    NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForScale, \
    NonStationaryAltitudinalScaleQuadraticCrossTermForScale
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models_with_scale import \
    NonStationaryAltitudinalScaleQuadraticCrossTermForLocation, NonStationaryAltitudinalScaleQuadratic, \
    NonStationaryAltitudinalLocationLinearScaleQuadratic, NonStationaryAltitudinalLocationQuadraticScaleQuadratic, \
    NonStationaryAltitudinalLocationLinearScaleQuadraticCrossTermForLocation, \
    NonStationaryAltitudinalLocationQuadraticScaleQuadraticCrossTermForLocation
from extreme_fit.model.margin_model.polynomial_margin_model.gumbel_altitudinal_models import \
    StationaryGumbelAltitudinal, NonStationaryGumbelAltitudinalScaleLinear, \
    NonStationaryGumbelAltitudinalLocationLinear, NonStationaryGumbelAltitudinalLocationQuadratic, \
    NonStationaryGumbelAltitudinalLocationLinearScaleLinear, NonStationaryGumbelAltitudinalLocationQuadraticScaleLinear, \
    NonStationaryGumbelAltitudinalLocationLinearCrossTermForLocation, \
    NonStationaryGumbelAltitudinalLocationQuadraticCrossTermForLocation, \
    NonStationaryGumbelAltitudinalLocationLinearScaleLinearCrossTermForLocation, \
    NonStationaryGumbelAltitudinalLocationQuadraticScaleLinearCrossTermForLocation, \
    NonStationaryGumbelAltitudinalScaleLinearCrossTermForLocation, NonStationaryGumbelCrossTermForLocation
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    NonStationaryLocationSpatioTemporalLinearityModel1, NonStationaryLocationSpatioTemporalLinearityModel2, \
    NonStationaryLocationSpatioTemporalLinearityModel3, NonStationaryLocationSpatioTemporalLinearityModel4, \
    NonStationaryLocationSpatioTemporalLinearityModel5, NonStationaryLocationSpatioTemporalLinearityModelAssertError1, \
    NonStationaryLocationSpatioTemporalLinearityModelAssertError2, \
    NonStationaryLocationSpatioTemporalLinearityModelAssertError3, NonStationaryLocationSpatioTemporalLinearityModel6


ALTITUDINAL_GEV_MODELS = [
                             StationaryAltitudinal,
                             NonStationaryAltitudinalScaleLinear,
                             NonStationaryAltitudinalLocationLinear,
                             NonStationaryAltitudinalLocationQuadratic,
                             NonStationaryAltitudinalLocationLinearScaleLinear,
                             NonStationaryAltitudinalLocationQuadraticScaleLinear,

                             NonStationaryCrossTermForLocation,
                             NonStationaryAltitudinalScaleLinearCrossTermForLocation,
                             NonStationaryAltitudinalLocationLinearCrossTermForLocation,
                             NonStationaryAltitudinalLocationQuadraticCrossTermForLocation,
                             NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForLocation,
                             NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation,

                             # Quadratic in the scale
                             # NonStationaryAltitudinalScaleQuadratic,
                             # NonStationaryAltitudinalLocationLinearScaleQuadratic,
                             # NonStationaryAltitudinalLocationQuadraticScaleQuadratic,
                             # NonStationaryAltitudinalScaleQuadraticCrossTermForLocation,
                             # NonStationaryAltitudinalLocationLinearScaleQuadraticCrossTermForLocation,
                             # NonStationaryAltitudinalLocationQuadraticScaleQuadraticCrossTermForLocation,
                             #
                             # # Cross term for the scale
                             # NonStationaryAltitudinalScaleLinearCrossTermForScale,
                             # NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForScale,
                             # NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForScale,
                             # NonStationaryAltitudinalScaleQuadraticCrossTermForScale,
                             # NonStationaryAltitudinalLocationLinearScaleQuadraticCrossTermForScale,
                             # NonStationaryAltitudinalLocationQuadraticScaleQuadraticCrossTermForScale,

                         ][:]


ALTITUDINAL_GUMBEL_MODELS = [
                                StationaryGumbelAltitudinal,
                                NonStationaryGumbelAltitudinalScaleLinear,
                                NonStationaryGumbelAltitudinalLocationLinear,
                                NonStationaryGumbelAltitudinalLocationQuadratic,
                                NonStationaryGumbelAltitudinalLocationLinearScaleLinear,
                                NonStationaryGumbelAltitudinalLocationQuadraticScaleLinear,

                                NonStationaryGumbelCrossTermForLocation,
                                NonStationaryGumbelAltitudinalLocationLinearCrossTermForLocation,
                                NonStationaryGumbelAltitudinalLocationQuadraticCrossTermForLocation,
                                NonStationaryGumbelAltitudinalLocationLinearScaleLinearCrossTermForLocation,
                                NonStationaryGumbelAltitudinalLocationQuadraticScaleLinearCrossTermForLocation,
                                NonStationaryGumbelAltitudinalScaleLinearCrossTermForLocation,
                            ][:]

MODELS_THAT_SHOULD_RAISE_AN_ASSERTION_ERROR = [NonStationaryLocationSpatioTemporalLinearityModelAssertError1,
                                               NonStationaryLocationSpatioTemporalLinearityModelAssertError2,
                                               NonStationaryLocationSpatioTemporalLinearityModelAssertError3]

VARIOUS_SPATIO_TEMPORAL_MODELS = [NonStationaryLocationSpatioTemporalLinearityModel1,
                                  NonStationaryLocationSpatioTemporalLinearityModel2,
                                  NonStationaryLocationSpatioTemporalLinearityModel3,
                                  NonStationaryLocationSpatioTemporalLinearityModel4,
                                  NonStationaryLocationSpatioTemporalLinearityModel5,
                                  NonStationaryLocationSpatioTemporalLinearityModel6,
                                  ]
