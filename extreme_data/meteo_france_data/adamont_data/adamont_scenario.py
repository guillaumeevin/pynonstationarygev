from enum import Enum

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import get_gcm_rcm_couple_adamont_to_full_name


class AdamontScenario(Enum):
    histo = 0
    rcp26 = 1
    rcp45 = 2
    rcp85 = 3
    rcp85_extended = 4


adamont_scenarios_real = [AdamontScenario.histo, AdamontScenario.rcp26, AdamontScenario.rcp45, AdamontScenario.rcp85]
rcp_scenarios = [AdamontScenario.rcp26, AdamontScenario.rcp45, AdamontScenario.rcp85]


def get_linestyle_from_scenario(adamont_scenario):
    assert isinstance(adamont_scenario, AdamontScenario)
    if adamont_scenario is AdamontScenario.rcp26:
        return 'dashed'
    elif adamont_scenario is AdamontScenario.rcp45:
        return 'dashdot'
    elif adamont_scenario is AdamontScenario.rcp85:
        return 'dotted'
    else:
        return 'solid'


def get_year_min_and_year_max_from_scenario(adamont_scenario, gcm_rcm_couple):
    assert isinstance(adamont_scenario, AdamontScenario)
    year_min = get_year_min(adamont_scenario, gcm_rcm_couple)
    year_max = get_year_max(adamont_scenario, gcm_rcm_couple)
    return year_min, year_max


def get_year_max(adamont_scenario, gcm_rcm_couple):
    real_adamont_scenarios = scenario_to_real_scenarios(adamont_scenario)
    gcm, rcm = gcm_rcm_couple
    if any([scenario is not AdamontScenario.histo for scenario in real_adamont_scenarios]):
        if gcm == 'HadGEM2-ES':
            year_max = 2099
        else:
            year_max = 2100
    else:
        year_max = 2005
    return year_max


def get_year_min(adamont_scenario, gcm_rcm_couple):
    real_adamont_scenarios = scenario_to_real_scenarios(adamont_scenario)
    gcm, rcm = gcm_rcm_couple
    if AdamontScenario.histo in real_adamont_scenarios:
        if gcm == 'HadGEM2-ES':
            year_min = 1982
        elif rcm == 'RCA4':
            year_min = 1971
        elif gcm_rcm_couple in [('NorESM1-M', 'HIRHAM5'), ('IPSL-CM5A-MR', 'WRF331F'), ('CNRM-CM5', 'ALADIN63'),
                                ('IPSL-CM5A-MR', 'WRF381P')]:
            year_min = 1952
        else:
            year_min = 1951
    else:
        year_min = 2006
    return year_min


def get_gcm_rcm_couple_adamont_version_2(scenario):
    s = set(list(get_gcm_rcm_couple_adamont_to_full_name(version=2).keys()))
    scenario_to_list_to_remove = {
        AdamontScenario.rcp26: [('EC-EARTH', 'CCLM4-8-17'), ('CNRM-CM5', 'ALADIN53'), ('CNRM-CM5', 'RCA4'),
                                ('MPI-ESM-LR', 'RCA4'), ('HadGEM2-ES', 'CCLM4-8-17'), ('IPSL-CM5A-MR', 'RCA4'),
                                ('CNRM-CM5', 'CCLM4-8-17'), ('IPSL-CM5A-MR', 'WRF381P'), ('NorESM1-M', 'HIRHAM5'),
                                ('IPSL-CM5A-MR', 'WRF331F'), ('HadGEM2-ES', 'RCA4')],
        AdamontScenario.rcp45:  [('NorESM1-M', 'REMO2015'), ('HadGEM2-ES', 'RegCM4-6')],
        AdamontScenario.rcp85: [],
    }
    for couple_to_remove in scenario_to_list_to_remove[scenario]:
        s.remove(couple_to_remove)
    return list(s)

def load_gcm_rcm_couples(year_min=None, year_max=None,
                         adamont_scenario=AdamontScenario.histo,
                         adamont_version=2):
    gcm_rcm_couples = []
    gcm_rcm_couple_to_full_name = get_gcm_rcm_couple_adamont_to_full_name(adamont_version)
    for gcm_rcm_couple in gcm_rcm_couple_to_full_name.keys():
        year_min_couple, year_max_couple = get_year_min_and_year_max_from_scenario(
            adamont_scenario=adamont_scenario,
            gcm_rcm_couple=gcm_rcm_couple)
        if (year_min is None) or (year_min_couple <= year_min):
            if (year_max is None) or (year_max <= year_max_couple):
                gcm_rcm_couples.append(gcm_rcm_couple)
    return gcm_rcm_couples


def get_suffix_for_the_nc_file(adamont_scenario, gcm_rcm_couple):
    assert isinstance(adamont_scenario, AdamontScenario)
    year_min, year_max = get_year_min_and_year_max_from_scenario(adamont_scenario, gcm_rcm_couple)
    return '{}080106_{}080106_daysum'.format(year_min - 1, year_max)


def scenario_to_str(adamont_scenario):
    return '+'.join([str(real_adamont_scenario).split('.')[-1].upper()
                     for real_adamont_scenario in scenario_to_real_scenarios(adamont_scenario)])


def scenario_to_real_scenarios(adamont_scenario):
    if adamont_scenario in adamont_scenarios_real:
        return [adamont_scenario]
    else:
        if adamont_scenario is AdamontScenario.rcp85_extended:
            return [AdamontScenario.histo, AdamontScenario.rcp85]
        else:
            raise NotImplementedError


def gcm_rcm_couple_to_str(gcm_rcm_couple):
    return ' / '.join(gcm_rcm_couple)

