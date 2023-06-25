import os.path as op

from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel, StationaryTemporalModel
from root_utils import get_root_path


mode_to_name = {
    0: 'piecewise_w',
    1: 'linear_w',
    2: 'piecewise_wo',
    3: 'linear_wo',
    4: 'constant_wo',
    5: 'constant_w',
}


def load_csv_filepath_gof(mode, altitude, all_massif):
    name = mode_to_name[mode]
    csv_filename = '{}_{}_{}'.format(name, altitude, int(all_massif))
    csv_filepath = op.join(get_root_path(), "gof", csv_filename + '.csv')
    return csv_filepath

def load_parameters(mode, massif_name_to_model_class, massif_name_to_parametrization_number):
    assert mode in mode_to_name
    # Force linear models
    if mode in [1, 3]:
        massif_name_to_model_class = {k: NonStationaryLocationAndScaleAndShapeTemporalModel
                                      for k in massif_name_to_model_class.keys()}
    # Force constant models
    if mode in [4, 5]:
        massif_name_to_model_class = {k: StationaryTemporalModel
                                      for k in massif_name_to_model_class.keys()}
    # For without adjsutement coefficients
    if mode in [2, 3, 4]:
        massif_name_to_parametrization_number = {k: 0 for k in massif_name_to_parametrization_number.keys()}
    return massif_name_to_model_class, massif_name_to_parametrization_number
