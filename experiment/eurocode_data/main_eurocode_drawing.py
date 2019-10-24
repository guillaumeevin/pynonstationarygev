import time
from collections import OrderedDict

from experiment.eurocode_data.eurocode_visualizer import plot_model_name_to_dep_to_ordered_return_level_uncertainties
from experiment.eurocode_data.massif_name_to_departement import DEPARTEMENT_TYPES
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


def dep_to_ordered_return_level_uncertainties(model_class, last_year_for_the_data, altitudes):
    model_type = get_display_name_from_object_type(model_class).split('Stationary')[0] + 'Stationary'
    # model_name += ' 1958-' + str(last_year_for_the_data)
    is_non_stationary = model_type == 'NonStationary'
    model_symbol = '{\mu_1, \sigma_1}' if is_non_stationary else '0'
    parameter = ', 2017' if is_non_stationary else ''
    model_name = ' $ \widehat{q_{\\textrm{GEV}}(\\boldsymbol{\\theta_{\mathcal{M}_'
    model_name += model_symbol
    model_name += '}}'
    model_name += parameter
    model_name += ')}_{ \\textrm{MMSE}} $ ' + '({})'.format(model_type)
    # Load altitude visualizer
    altitude_visualizer = load_altitude_visualizer(AltitudeHypercubeVisualizer, altitudes=altitudes,
                                                   last_starting_year=None, nb_data_reduced_for_speed=False,
                                                   only_first_one=False, save_to_file=False,
                                                   exact_starting_year=1958,
                                                   first_starting_year=None,
                                                   study_classes=[CrocusSwe3Days],
                                                   trend_test_class=None)
    # Loop on the data
    assert isinstance(altitude_visualizer.tuple_to_study_visualizer, OrderedDict)
    dep_to_ordered_return_level_uncertainty = {dep: [] for dep in DEPARTEMENT_TYPES}
    for altitude, visualizer in altitude_visualizer.tuple_to_study_visualizer.items():
        print('{} processing altitude = {} '.format(model_name, altitude))
        dep_to_return_level_uncertainty = visualizer.dep_class_to_eurocode_level_uncertainty(model_class,
                                                                                             last_year_for_the_data)
        for dep, return_level_uncertainty in dep_to_return_level_uncertainty.items():
            dep_to_ordered_return_level_uncertainty[dep].append(return_level_uncertainty)

    return {model_name: dep_to_ordered_return_level_uncertainty}


def main_drawing():
    # Select parameters
    fast_plot = [True, False][1]
    model_class_and_last_year = [
                                    (StationaryTemporalModel, LAST_YEAR_FOR_EUROCODE),
                                    (StationaryTemporalModel, 2017),
                                    (NonStationaryLocationAndScaleTemporalModel, 2017),
                                ][1:]
    altitudes = EUROCODE_ALTITUDES[:]
    if fast_plot:
        model_class_and_last_year = model_class_and_last_year[:2]
        altitudes = altitudes[:2]

    model_name_to_dep_to_ordered_return_level = {}
    for model_class, last_year_for_the_data in model_class_and_last_year:
        start = time.time()
        model_name_to_dep_to_ordered_return_level.update(
            dep_to_ordered_return_level_uncertainties(model_class, last_year_for_the_data, altitudes))
        duration = time.time() - start
        print(model_class, duration)
    # Transform the dictionary into the desired format
    dep_to_model_name_to_ordered_return_level_uncertainties = {}
    for dep in DEPARTEMENT_TYPES:
        d2 = {model_name: model_name_to_dep_to_ordered_return_level[model_name][dep] for model_name in
              model_name_to_dep_to_ordered_return_level.keys()}
        dep_to_model_name_to_ordered_return_level_uncertainties[dep] = d2
    # Plot graph
    plot_model_name_to_dep_to_ordered_return_level_uncertainties(
        dep_to_model_name_to_ordered_return_level_uncertainties, altitudes, show=True)


if __name__ == '__main__':
    main_drawing()
