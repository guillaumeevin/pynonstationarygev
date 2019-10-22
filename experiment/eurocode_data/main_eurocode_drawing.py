from collections import OrderedDict

from experiment.eurocode_data.eurocode_visualizer import plot_model_name_to_dep_to_ordered_return_level_uncertainties
from experiment.eurocode_data.massif_name_to_departement import DEPARTEMENT_TYPES
from experiment.eurocode_data.utils import EUROCODE_ALTITUDES, LAST_YEAR_FOR_EUROCODE
from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSwe3Days
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.utils_hypercube import \
    load_altitude_visualizer
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryStationModel, \
    NonStationaryLocationAndScaleModel
from root_utils import get_display_name_from_object_type


# Model class


def dep_to_ordered_return_level_uncertainties(model_class, last_year_for_the_data):
    model_name = get_display_name_from_object_type(type(model_class)) + ' ' + str(last_year_for_the_data)
    # Load altitude visualizer
    # todo: add last years attributes that enables to change the years
    altitude_visualizer = load_altitude_visualizer(AltitudeHypercubeVisualizer, altitudes=EUROCODE_ALTITUDES,
                                                   last_starting_year=None, nb_data_reduced_for_speed=False,
                                                   only_first_one=False, save_to_file=False,
                                                   exact_starting_year=1958,
                                                   first_starting_year=None,
                                                   study_classes=[CrocusSwe3Days],
                                                   trend_test_class=None)
    # Loop on the data
    assert isinstance(altitude_visualizer.tuple_to_study_visualizer, OrderedDict)
    dep_to_ordered_return_level_uncertainty = {dep: [] for dep in DEPARTEMENT_TYPES}
    for visualizer in altitude_visualizer.tuple_to_study_visualizer.values():
        dep_to_return_level_uncertainty = visualizer.dep_class_to_eurocode_level_uncertainty(model_class, last_year_for_the_data)
        for dep, return_level_uncertainty in dep_to_return_level_uncertainty.items():
            dep_to_ordered_return_level_uncertainty[dep].append(return_level_uncertainty)

    return {model_name: dep_to_ordered_return_level_uncertainty}


def main_drawing():
    model_class_and_last_year = [
        (StationaryStationModel, LAST_YEAR_FOR_EUROCODE),
        (StationaryStationModel, 2017),
        (NonStationaryLocationAndScaleModel, 2017),
    ][:1]
    model_name_to_dep_to_ordered_return_level = {}
    for model_class, last_year_for_the_data in model_class_and_last_year:
        model_name_to_dep_to_ordered_return_level.update(
            dep_to_ordered_return_level_uncertainties(model_class, last_year_for_the_data))
    # Transform the dictionary into the desired format
    dep_to_model_name_to_ordered_return_level_uncertainties = {}
    for dep in DEPARTEMENT_TYPES:
        d2 = {model_name: model_name_to_dep_to_ordered_return_level[model_name][dep] for model_name in
              model_name_to_dep_to_ordered_return_level.keys()}
        dep_to_model_name_to_ordered_return_level_uncertainties[dep] = d2
    # Plot graph
    plot_model_name_to_dep_to_ordered_return_level_uncertainties(
        dep_to_model_name_to_ordered_return_level_uncertainties, show=True)


if __name__ == '__main__':
    main_drawing()
