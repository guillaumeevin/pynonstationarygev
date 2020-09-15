from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal, \
    NonStationaryAltitudinalLocationLinear, NonStationaryAltitudinalLocationQuadratic, \
    NonStationaryAltitudinalLocationLinearScaleLinear, NonStationaryAltitudinalLocationQuadraticScaleLinear, \
    NonStationaryCrossTermForLocation, NonStationaryAltitudinalLocationLinearCrossTermForLocation, \
    NonStationaryAltitudinalLocationQuadraticCrossTermForLocation, \
    NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForLocation, \
    NonStationaryAltitudinalScaleLinear, \
    NonStationaryAltitudinalLocationCubicCrossTermForLocation, \
    NonStationaryAltitudinalLocationCubic
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models_only_altitude_and_scale import \
    StationaryAltitudinalOnlyScale, NonStationaryAltitudinalOnlyScaleLocationLinear, \
    NonStationaryAltitudinalOnlyScaleLocationQuadratic, NonStationaryAltitudinalOnlyScaleLocationCubic, \
    NonStationaryAltitudinalOnlyScaleLocationOrder4, NonStationaryOnlyScaleCrossTermForLocation, \
    NonStationaryAltitudinalOnlyScaleLocationLinearCrossTermForLocation, \
    NonStationaryAltitudinalOnlyScaleLocationQuadraticCrossTermForLocation, \
    NonStationaryAltitudinalOnlyScaleLocationCubicCrossTermForLocation, \
    NonStationaryAltitudinalOnlyScaleLocationOrder4CrossTermForLocation
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models_with_scale_2 import \
    NonStationaryAltitudinalScaleLinearCrossTermForLocation, \
    NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation, \
    NonStationaryAltitudinalLocationCubicScaleLinearCrossTermForLocation, \
    NonStationaryAltitudinalLocationCubicScaleLinear
from extreme_fit.model.margin_model.polynomial_margin_model.gumbel_altitudinal_models import \
    StationaryGumbelAltitudinal, NonStationaryGumbelAltitudinalScaleLinear, \
    NonStationaryGumbelAltitudinalLocationLinear, NonStationaryGumbelAltitudinalLocationQuadratic, \
    NonStationaryGumbelAltitudinalLocationLinearScaleLinear, NonStationaryGumbelAltitudinalLocationQuadraticScaleLinear, \
    NonStationaryGumbelAltitudinalLocationLinearCrossTermForLocation, \
    NonStationaryGumbelAltitudinalLocationQuadraticCrossTermForLocation, \
    NonStationaryGumbelAltitudinalLocationLinearScaleLinearCrossTermForLocation, \
    NonStationaryGumbelAltitudinalLocationQuadraticScaleLinearCrossTermForLocation, \
    NonStationaryGumbelAltitudinalScaleLinearCrossTermForLocation, NonStationaryGumbelCrossTermForLocation
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_quadratic_location_and_scale_with_constant_shape import \
    AltitudinalShapeConstantTimeLocationQuadratic, AltitudinalShapeConstantTimeLocationQuadraticScaleLinear, \
    AltitudinalShapeConstantTimeLocationQuadraticScaleQuadratic, \
    AltitudinalShapeConstantTimeLocationLinearScaleQuadratic, AltitudinalShapeConstantTimeScaleQuadratic
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_quadratic_location_and_scale_with_linear_shape import \
    AltitudinalShapeLinearTimeLocationQuadratic, AltitudinalShapeLinearTimeLocationQuadraticScaleLinear, \
    AltitudinalShapeLinearTimeLocationQuadraticScaleQuadratic, AltitudinalShapeLinearTimeLocationLinearScaleQuadratic, \
    AltitudinalShapeLinearTimeScaleQuadratic
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_constant_shape_wrt_altitude import \
    AltitudinalShapeConstantTimeLocShapeLinear, \
    AltitudinalShapeConstantTimeLocScaleLinear, AltitudinalShapeConstantTimeScaleShapeLinear, \
    AltitudinalShapeConstantTimeLocScaleShapeLinear, AltitudinalShapeConstantTimeLocationLinear, \
    AltitudinalShapeConstantTimeScaleLinear, AltitudinalShapeConstantTimeShapeLinear
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_linear_shape_wrt_altitude import \
    AltitudinalShapeLinearTimeStationary, AltitudinalShapeLinearTimeLocScaleShapeLinear, \
    AltitudinalShapeLinearTimeLocationLinear, AltitudinalShapeLinearTimeScaleLinear, \
    AltitudinalShapeLinearTimeShapeLinear, AltitudinalShapeLinearTimeLocShapeLinear, \
    AltitudinalShapeLinearTimeLocScaleLinear, AltitudinalShapeLinearTimeScaleShapeLinear
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    NonStationaryLocationSpatioTemporalLinearityModel1, NonStationaryLocationSpatioTemporalLinearityModel2, \
    NonStationaryLocationSpatioTemporalLinearityModel3, NonStationaryLocationSpatioTemporalLinearityModel4, \
    NonStationaryLocationSpatioTemporalLinearityModel5, NonStationaryLocationSpatioTemporalLinearityModelAssertError1, \
    NonStationaryLocationSpatioTemporalLinearityModelAssertError2, \
    NonStationaryLocationSpatioTemporalLinearityModelAssertError3, NonStationaryLocationSpatioTemporalLinearityModel6

ALTITUDINAL_GEV_MODELS_LOCATION = [
    StationaryAltitudinal,
    # NonStationaryAltitudinalLocationLinear,
    # NonStationaryAltitudinalLocationQuadratic,
    # NonStationaryAltitudinalLocationCubic,
    # NonStationaryAltitudinalLocationOrder4,
    # # First order cross term
    # NonStationaryCrossTermForLocation,

    # NonStationaryAltitudinalLocationLinearCrossTermForLocation,
    NonStationaryAltitudinalLocationQuadraticCrossTermForLocation,
    NonStationaryAltitudinalLocationCubicCrossTermForLocation,
    # NonStationaryAltitudinalLocationOrder4CrossTermForLocation,
    # NonStationaryAltitudinalScaleLinearCrossTermForLocation,
    # NonStationaryAltitudinalScaleLinearLocationLinearCrossTermForLocation,

    # NonStationaryAltitudinalScaleQuadraticCrossTermForLocation,
    # NonStationaryAltitudinalScaleQuadraticLocationLinearCrossTermForLocation,

    # NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation,
    # NonStationaryAltitudinalLocationCubicScaleLinearCrossTermForLocation,
    # NonStationaryAltitudinalLocationQuadraticScaleQuadraticCrossTermForLocation,
    # NonStationaryAltitudinalLocationCubicScaleQuadraticCrossTermForLocation,

]
ALTITUDINAL_GEV_MODELS_LOCATION_QUADRATIC_MINIMUM = [
    StationaryAltitudinal,
    # NonStationaryAltitudinalLocationLinear,
    NonStationaryAltitudinalLocationQuadratic,
    NonStationaryAltitudinalLocationCubic,
    # NonStationaryAltitudinalLocationOrder4,
    # # First order cross term
    # NonStationaryCrossTermForLocation,
    # NonStationaryAltitudinalLocationLinearCrossTermForLocation,
    NonStationaryAltitudinalLocationQuadraticCrossTermForLocation,
    NonStationaryAltitudinalLocationCubicCrossTermForLocation,
    # NonStationaryAltitudinalLocationOrder4CrossTermForLocation,

    NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation,
    NonStationaryAltitudinalLocationQuadraticScaleLinear,
    NonStationaryAltitudinalLocationCubicScaleLinear,
    NonStationaryAltitudinalLocationCubicScaleLinearCrossTermForLocation,

]

ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS = [

    # With a constant shape for the altitude (8 models)
    StationaryAltitudinal,
    AltitudinalShapeConstantTimeLocationLinear,
    AltitudinalShapeConstantTimeScaleLinear,
    # AltitudinalShapeConstantTimeShapeLinear,
    # AltitudinalShapeConstantTimeLocShapeLinear,
    AltitudinalShapeConstantTimeLocScaleLinear,
    # AltitudinalShapeConstantTimeScaleShapeLinear,
    # AltitudinalShapeConstantTimeLocScaleShapeLinear,
    # With a linear shape for the altitude (8 models)
    AltitudinalShapeLinearTimeStationary,
    AltitudinalShapeLinearTimeLocationLinear,
    AltitudinalShapeLinearTimeScaleLinear,
    # AltitudinalShapeLinearTimeShapeLinear,
    # AltitudinalShapeLinearTimeLocShapeLinear,
    AltitudinalShapeLinearTimeLocScaleLinear,
    # AltitudinalShapeLinearTimeScaleShapeLinear,
    # AltitudinalShapeLinearTimeLocScaleShapeLinear,

    # With a quadratic term

    # AltitudinalShapeConstantTimeLocationQuadratic,
    # AltitudinalShapeConstantTimeLocationQuadraticScaleLinear,
    # AltitudinalShapeConstantTimeLocationQuadraticScaleQuadratic,
    # AltitudinalShapeConstantTimeLocationLinearScaleQuadratic,
    # AltitudinalShapeConstantTimeScaleQuadratic,
    #
    # AltitudinalShapeLinearTimeLocationQuadratic,
    # AltitudinalShapeLinearTimeLocationQuadraticScaleLinear,
    # AltitudinalShapeLinearTimeLocationQuadraticScaleQuadratic,
    # AltitudinalShapeLinearTimeLocationLinearScaleQuadratic,
    # AltitudinalShapeLinearTimeScaleQuadratic,

]

ALTITUDINAL_GEV_MODELS_LOCATION_ONLY_SCALE_ALTITUDES = [
    StationaryAltitudinalOnlyScale,
    NonStationaryAltitudinalOnlyScaleLocationLinear,
    NonStationaryAltitudinalOnlyScaleLocationQuadratic,
    NonStationaryAltitudinalOnlyScaleLocationCubic,
    NonStationaryAltitudinalOnlyScaleLocationOrder4,
    # Cross terms
    NonStationaryOnlyScaleCrossTermForLocation,
    NonStationaryAltitudinalOnlyScaleLocationLinearCrossTermForLocation,
    NonStationaryAltitudinalOnlyScaleLocationQuadraticCrossTermForLocation,
    NonStationaryAltitudinalOnlyScaleLocationCubicCrossTermForLocation,
    NonStationaryAltitudinalOnlyScaleLocationOrder4CrossTermForLocation,
]

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
