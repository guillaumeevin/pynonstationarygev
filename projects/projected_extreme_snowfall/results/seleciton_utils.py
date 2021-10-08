from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationAndScaleAndShapeTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel, NonStationaryThreeLinearLocationAndScaleAndShapeModel, \
    NonStationaryFourLinearLocationAndScaleAndShapeModel

number_to_model_name = {
    0: "StationaryTemporalModel",
    1: "NonStationaryLocationAndScaleAndShapeTemporalModel",
    2: "NonStationaryTwoLinearLocationAndScaleAndShapeModel",
    3: "NonStationaryThreeLinearLocationAndScaleAndShapeModel",
    4: "NonStationaryFourLinearLocationAndScaleAndShapeModel",
    5: "NonStationaryFiveLinearLocationAndScaleAndShapeModel",
    6: "NonStationarySixLinearLocationAndScaleAndShapeModel",
    7: "NonStationarySevenLinearLocationAndScaleAndShapeModel",
    8: "NonStationaryEightLinearLocationAndScaleAndShapeModel",
    9: "NonStationaryNineLinearLocationAndScaleAndShapeModel",
    10: "NonStationaryTenLinearLocationAndScaleAndShapeModel",
}

linear_effects_for_selection = (False, False, False)

number_to_model_class = {
    0: StationaryTemporalModel,
    1: NonStationaryLocationAndScaleAndShapeTemporalModel,
    2: NonStationaryTwoLinearLocationAndScaleAndShapeModel,
    3: NonStationaryThreeLinearLocationAndScaleAndShapeModel,
    4: NonStationaryFourLinearLocationAndScaleAndShapeModel,
}

model_class_to_number = {v: k for k, v in number_to_model_class.items()}

short_name_to_parametrization_number = {
    "no effect": 0,
    "is_ensemble_member": 5,
    "gcm": 1,
    "rcm": 2,
    "gcm_and_rcm": 4,
}

parametrization_number_to_short_name = {
    v:k for k,v in short_name_to_parametrization_number.items()
}

short_name_to_color = {
    "without obs": "grey",
    "no effect": "paleturquoise",
    # "is_ensemble_member": 'lemonchiffon',
    # "is_ensemble_member": 'beige',
    "is_ensemble_member": 'palegoldenrod',
    "gcm": 'navajowhite',
    "rcm": "lightsalmon",
    "gcm_and_rcm": 'plum',
    "mean": "black",
}

short_name_to_label = {
    'without obs': "Baseline",
    "no effect": "Zero adjustment coefficient",
    "gcm": 'One adjustment coefficient for each GCM',
    "gcm_and_rcm": 'One adjustment coefficient for each GCM-RCM pair',
    "is_ensemble_member": 'One adjustment coefficient for all GCM-RCM pairs',
    "rcm": "One adjustment coefficient for each RCM",
    "mean": "Average on the 5 parameterization",
}


def get_short_name(i):
    prefix, short_name = i.split("Obs_with obs and ")
    if 'loc' in short_name:
        short_name = short_name.split()[0][4:]
    return short_name

