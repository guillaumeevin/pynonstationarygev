import os.path as op

from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel, StationaryTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleOneLinearShapeModel, \
    NonStationaryThreeLinearLocationAndScaleOneLinearShapeModel, \
    NonStationaryFourLinearLocationAndScaleOneLinearShapeModel
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
    9: 'two_wo',
    10: 'three_wo',
    11: 'four_wo',
}


def load_csv_filepath_gof(mode, altitude, all_massif):
    name = mode_to_name[mode]
    csv_filename = '{}_{}_{}'.format(name, altitude, int(all_massif))
    csv_filepath = op.join(get_root_path(), "data", "gof", csv_filename + '.csv')
    return csv_filepath

def load_parameters(mode, massif_name_to_model_class, massif_name_to_parametrization_number):
    assert mode in mode_to_name
    # Force exact models
    new_model = None
    if mode == 9:
        new_model = NonStationaryTwoLinearLocationAndScaleOneLinearShapeModel
    if mode == 10:
        new_model = NonStationaryThreeLinearLocationAndScaleOneLinearShapeModel
    if mode == 11:
        new_model = NonStationaryFourLinearLocationAndScaleOneLinearShapeModel

    # Force linear models
    if mode in [1, 3, 7]:
        new_model = NonStationaryLocationAndScaleAndShapeTemporalModel
    # Force constant models
    if mode in [4, 5, 8]:
        new_model = StationaryTemporalModel
    # For without adjsutement coefficients
    if mode in [2, 3, 4, 9, 10, 11]:
        massif_name_to_parametrization_number = {k: 0 for k in massif_name_to_parametrization_number.keys()}
    if mode in [6, 7, 8]:
        massif_name_to_parametrization_number = {k: 5 for k in massif_name_to_parametrization_number.keys()}

    if new_model is not None:
        massif_name_to_model_class = {k: new_model
                                      for k in massif_name_to_model_class.keys()}
    return massif_name_to_model_class, massif_name_to_parametrization_number
