import itertools

from extreme_fit.distribution.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates

climate_coordinates_with_effects_list = [None,
                                         [AbstractCoordinates.COORDINATE_GCM],
                                         [AbstractCoordinates.COORDINATE_RCM],
                                         [AbstractCoordinates.COORDINATE_GCM, AbstractCoordinates.COORDINATE_RCM],
                                         [AbstractCoordinates.COORDINATE_GCM_AND_RCM],
                                         ]  # None means we do not create any effect


def load_combination_name_for_tuple(combination):
    return load_combination_name(load_param_name_to_climate_coordinates_with_effects(combination))


def load_param_name_to_climate_coordinates_with_effects(combination):
    param_name_to_climate_coordinates_with_effects = {param_name: climate_coordinates_with_effects_list[idx]
                                                      for param_name, idx in
                                                      zip(GevParams.PARAM_NAMES, combination)}
    if all([v is None for v in param_name_to_climate_coordinates_with_effects.values()]):
        return None
    else:
        return param_name_to_climate_coordinates_with_effects


def generate_sub_combination(combination):
    number_to_sub_numbers = {
        0: [0],
        1: [1],
        2: [2],
        3: [1, 2, 3],
        4: [4],
    }
    return list(itertools.product(*[number_to_sub_numbers[number] for number in combination]))

def load_combination(param_name_to_climate_coordinates_with_effects):
    if param_name_to_climate_coordinates_with_effects is None:
        return 0, 0, 0
    else:
        def transform(e):
            return tuple(e) if isinstance(e, list) else None
        effects_list = [transform(e) for e in climate_coordinates_with_effects_list]
        combination = []
        for param_name in GevParams.PARAM_NAMES:
            number = effects_list.index(transform(param_name_to_climate_coordinates_with_effects[param_name]))
            combination.append(number)
        return tuple(combination)


def load_combination_name(param_name_to_climate_coordinates_with_effects):
    if param_name_to_climate_coordinates_with_effects is None:
        combination_name = 'no effect'
    else:

        param_name_to_effect_name = {p: '+'.join([e.replace('coord_', '') for e in l])
                                     for p, l in param_name_to_climate_coordinates_with_effects.items() if
                                     l is not None}
        combination_name = ' '.join(
            [param_name + '_' + param_name_to_effect_name[param_name] for param_name in GevParams.PARAM_NAMES
             if param_name in param_name_to_effect_name])
    return combination_name
