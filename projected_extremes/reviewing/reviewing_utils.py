import os.path as op

from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel, StationaryTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleOneLinearShapeModel, \
    NonStationaryThreeLinearLocationAndScaleOneLinearShapeModel, \
    NonStationaryFourLinearLocationAndScaleOneLinearShapeModel, NonStationaryTwoLinearLocationAndScaleAndShapeModel, \
    NonStationaryThreeLinearLocationAndScaleAndShapeModel, NonStationaryFourLinearLocationAndScaleAndShapeModel
from root_utils import get_root_path


mode_to_name = {
    0: 'piecewise_w',
    1: 'linear_w',
    2: 'piecewise_wo',
    3: 'linear_wo',
    4: 'constant_wo',
    5: 'constant_w',
    6: 'piecewise_fulleffect',
    7: 'linear_fulleffect',
    8: 'constant_fulleffect',
    9: 'one_fullfull',
    10: 'two_fullfull',
    11: 'three_fullfull',
    12: 'four_fullfull',
    13: 'selected_fullfull',
}


def load_csv_filepath_gof(mode, altitude, all_massif):
    name = mode_to_name[mode]
    csv_filename = '{}_{}_{}'.format(name, altitude, int(all_massif))
    csv_filepath = op.join(get_root_path(), "data", "gof", csv_filename + '.csv')
    return csv_filepath

def load_parameters(mode, massif_name_to_model_class, massif_name_to_parametrization_number):
    assert mode in mode_to_name
    new_model = None
    new_parametrization_number = None
    # For exact model
    if mode == 9:
        new_model = NonStationaryLocationAndScaleAndShapeTemporalModel
    if mode == 10:
        new_model = NonStationaryTwoLinearLocationAndScaleAndShapeModel
    if mode == 11:
        new_model = NonStationaryThreeLinearLocationAndScaleAndShapeModel
    if mode == 12:
        new_model = NonStationaryFourLinearLocationAndScaleAndShapeModel
    # Force linear models
    if mode in [1, 3, 7, 10]:
        new_model = NonStationaryLocationAndScaleAndShapeTemporalModel
    # Force constant models
    if mode in [4, 5, 8]:
        new_model = StationaryTemporalModel
    # For without adjsutement coefficients
    if mode in [2, 3, 4]:
        new_parametrization_number = 0
    if mode in [6, 7, 8]:
        new_parametrization_number = 5
    if mode in [9, 10, 11, 12, 13]:
        new_parametrization_number = 4

    if new_model is not None:
        massif_name_to_model_class = {k: new_model
                                      for k in massif_name_to_model_class.keys()}
    if new_parametrization_number is not None:
        massif_name_to_parametrization_number = {k: new_parametrization_number for k in massif_name_to_parametrization_number.keys()}

    return massif_name_to_model_class, massif_name_to_parametrization_number
