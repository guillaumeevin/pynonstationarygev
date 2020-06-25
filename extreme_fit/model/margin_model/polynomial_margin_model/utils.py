from extreme_fit.model.margin_model.polynomial_margin_model.altitudinal_models import StationaryAltitudinal, \
    NonStationaryAltitudinalLocationLinear, NonStationaryAltitudinalLocationQuadratic, \
    NonStationaryAltitudinalLocationLinearScaleLinear, NonStationaryAltitudinalLocationQuadraticScaleLinear, \
    NonStationaryCrossTermForLocation, NonStationaryAltitudinalLocationLinearCrossTermForLocation, \
    NonStationaryAltitudinalLocationQuadraticCrossTermForLocation, \
    NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForLocation, \
    NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    NonStationaryLocationSpatioTemporalLinearityModel1, NonStationaryLocationSpatioTemporalLinearityModel2, \
    NonStationaryLocationSpatioTemporalLinearityModel3, NonStationaryLocationSpatioTemporalLinearityModel4, \
    NonStationaryLocationSpatioTemporalLinearityModel5, NonStationaryLocationSpatioTemporalLinearityModelAssertError1, \
    NonStationaryLocationSpatioTemporalLinearityModelAssertError2, \
    NonStationaryLocationSpatioTemporalLinearityModelAssertError3, NonStationaryLocationSpatioTemporalLinearityModel6
ALTITUDINAL_MODELS = [
                         StationaryAltitudinal,
                         NonStationaryAltitudinalLocationLinear,
                         NonStationaryAltitudinalLocationQuadratic,
                         NonStationaryAltitudinalLocationLinearScaleLinear,
                         NonStationaryAltitudinalLocationQuadraticScaleLinear,

                         NonStationaryCrossTermForLocation,
                         NonStationaryAltitudinalLocationLinearCrossTermForLocation,
                         NonStationaryAltitudinalLocationQuadraticCrossTermForLocation,
                         NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForLocation,
                        NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation,
                     ][:7]




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
