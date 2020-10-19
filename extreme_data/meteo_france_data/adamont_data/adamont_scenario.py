from enum import Enum


class AdamontScenario(Enum):
    histo = 0
    rcp26 = 1
    rcp45 = 2
    rcp85 = 3


def get_year_min_and_year_max_from_scenario(adamont_scenario, gcm_rcm_couple):
    assert isinstance(adamont_scenario, AdamontScenario)
    if adamont_scenario == AdamontScenario.histo:
        gcm, rcm = gcm_rcm_couple
        if gcm == 'HadGEM2-ES':
            year_min = 1982
        elif rcm == 'RCA4':
            year_min = 1971
        elif gcm_rcm_couple in [('NorESM1-M', 'DMI-HIRHAM5'), ('IPSL-CM5A-MR', 'WRF331F')]:
            year_min = 1952
        else:
            year_min = 1951
        return year_min, 2005
    else:
        return 2006, 2100


def get_suffix_for_the_nc_file(adamont_scenario, gcm_rcm_couple):
    year_min, year_max = get_year_min_and_year_max_from_scenario(adamont_scenario, gcm_rcm_couple)
    return '{}080106_{}080106_daysum'.format(year_min-1, year_max)


def scenario_to_str(adamont_scenario):
    return str(adamont_scenario).split('.')[-1].upper()


gcm_rcm_couple_to_full_name = {
    ('CNRM-CM5', 'ALADIN53'): 'CNRM-ALADIN53_CNRM-CERFACS-CNRM-CM5',
    ('CNRM-CM5', 'RCA4'): 'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5',
    ('CNRM-CM5', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5',
    ('EC-EARTH', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_ICHEC-EC-EARTH',
    ('MPI-ESM-LR', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR',
    ('HadGEM2-ES', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
    ('HadGEM2-ES', 'RACMO22E'): 'KNMI-RACMO22E_MOHC-HadGEM2-ES',

    ('NorESM1-M', 'DMI-HIRHAM5'): 'DMI-HIRHAM5_NCC-NorESM1-M',
    ('IPSL-CM5A-MR', 'WRF331F'): 'IPSL-INERIS-WRF331F_IPSL-IPSL-CM5A-MR',
    ('EC-EARTH', 'RCA4'): 'SMHI-RCA4_ICHEC-EC-EARTH',
    ('IPSL-CM5A-MR', 'RCA4'): 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
    ('HadGEM2-ES', 'RCA4'): 'SMHI-RCA4_MOHC-HadGEM2-ES',
    ('MPI-ESM-LR', 'RCA4'): 'SMHI-RCA4_MPI-M-MPI-ESM-LR',


    ('MPI-ESM-LR', 'REMO2009'): 'MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',


}
