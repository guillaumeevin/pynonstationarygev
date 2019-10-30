import time
from collections import OrderedDict

from experiment.eurocode_data.eurocode_return_level_uncertainties import ConfidenceIntervalMethodFromExtremes
from experiment.eurocode_data.eurocode_visualizer import \
    plot_massif_name_to_model_name_to_uncertainty_method_to_ordered_dict, get_model_name
from experiment.eurocode_data.massif_name_to_departement import DEPARTEMENT_TYPES, MASSIF_NAMES_ALPS
from experiment.eurocode_data.utils import EUROCODE_ALTITUDES, LAST_YEAR_FOR_EUROCODE
from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSwe3Days
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.utils_hypercube import \
    load_altitude_visualizer
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationAndScaleTemporalModel
from root_utils import get_display_name_from_object_type

# Model class
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def massif_name_to_ordered_return_level_uncertainties(model_class, last_year_for_the_data, altitudes, massif_names, uncertainty_methods):
    # Load model name
    model_name = get_model_name(model_class)
    # Load altitude visualizer
    altitude_visualizer = load_altitude_visualizer(AltitudeHypercubeVisualizer, altitudes=altitudes,
                                                   last_starting_year=None, nb_data_reduced_for_speed=False,
                                                   only_first_one=False, save_to_file=False,
                                                   exact_starting_year=1958,
                                                   first_starting_year=None,
                                                   study_classes=[CrocusSwe3Days],
                                                   trend_test_class=None)  # type: AltitudeHypercubeVisualizer
    # Loop on the data
    assert isinstance(altitude_visualizer.tuple_to_study_visualizer, OrderedDict)
    massif_name_to_ordered_eurocode_level_uncertainty = {massif_name: {ci_method: [] for ci_method in uncertainty_methods} for massif_name in massif_names}
    for altitude, visualizer in altitude_visualizer.tuple_to_study_visualizer.items():
        print('{} processing altitude = {} '.format(model_name, altitude))
        for ci_method in uncertainty_methods:
            d = visualizer.massif_name_to_altitude_and_eurocode_level_uncertainty(model_class, last_year_for_the_data, massif_names, ci_method)
            # Append the altitude one by one
            for massif_name, return_level_uncertainty in d.items():
                massif_name_to_ordered_eurocode_level_uncertainty[massif_name][ci_method].append(return_level_uncertainty)
    return {model_name: massif_name_to_ordered_eurocode_level_uncertainty}





def main_drawing():
    fast_plot = [True, False][0]
    # Select parameters
    massif_names = MASSIF_NAMES_ALPS[:]
    model_class_and_last_year = [
                                    (StationaryTemporalModel, LAST_YEAR_FOR_EUROCODE),
                                    (StationaryTemporalModel, 2017),
                                    (NonStationaryLocationAndScaleTemporalModel, 2017),
                                ][1:]
    altitudes = EUROCODE_ALTITUDES[:]
    uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.my_bayes, ConfidenceIntervalMethodFromExtremes.bayes]

    if fast_plot:
        model_class_and_last_year = model_class_and_last_year[:1]
        altitudes = altitudes[2:4]
        massif_names = massif_names[:1]
        uncertainty_methods = uncertainty_methods[:1]

    model_name_to_massif_name_to_ordered_return_level = {}
    for model_class, last_year_for_the_data in model_class_and_last_year:
        start = time.time()
        model_name_to_massif_name_to_ordered_return_level.update(
            massif_name_to_ordered_return_level_uncertainties(model_class, last_year_for_the_data, altitudes, massif_names, uncertainty_methods))
        duration = time.time() - start
        print(model_class, duration)
    # Transform the dictionary into the desired format
    massif_name_to_model_name_to_ordered_return_level_uncertainties = {}
    for massif_name in massif_names:
        d2 = {model_name: model_name_to_massif_name_to_ordered_return_level[model_name][massif_name] for model_name in
              model_name_to_massif_name_to_ordered_return_level.keys()}
        massif_name_to_model_name_to_ordered_return_level_uncertainties[massif_name] = d2
    # Plot graph
    plot_massif_name_to_model_name_to_uncertainty_method_to_ordered_dict(
        massif_name_to_model_name_to_ordered_return_level_uncertainties, nb_massif_names=len(massif_names),
        nb_model_names=len(model_class_and_last_year), show=True)


if __name__ == '__main__':
    main_drawing()
