def get_year_min_and_year_max_used_to_compute_quantile(gcm):
    """pour les périodes de calcul de quantiles/quantiles, voici le recap :
    1980-2011 pour la réanalyse
    1974-2005 pour le modèle

    sauf GCM=MOHC-HadGEM2-ES
    1987-2011 pour la réanalyse
    1981-2005 pour le modèle
    """
    if gcm == 'HadGEM2-ES':
        reanalysis_years = (1988, 2011)
        model_year = (1982, 2005)
    else:
        reanalysis_years = (1981, 2011)
        model_year = (1975, 2005)
    return reanalysis_years, model_year


def get_rcm_gcm_couple_full_name(gcm_rcm_couple, adamont_version):
    return get_gcm_rcm_couple_adamont_to_full_name(adamont_version)[gcm_rcm_couple]


def get_gcm_rcm_couple_adamont_to_full_name(adamont_version):
    if adamont_version == 1:
        return _gcm_rcm_couple_adamont_v1_to_full_name
    else:
        return _gcm_rcm_couple_adamont_v2_to_full_name

gcm_to_rnumber = \
    {
        'MPI-ESM-LR': 1,
        'CNRM-CM5': 1,
        'IPSL-CM5A-MR': 1,
        'EC-EARTH': 12,
        'HadGEM2-ES': 1,
        'NorESM1-M': 1
    }

_gcm_rcm_couple_adamont_v1_to_full_name = {
    ('CNRM-CM5', 'ALADIN53'): 'CNRM-ALADIN53_CNRM-CERFACS-CNRM-CM5',
    ('CNRM-CM5', 'RCA4'): 'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5',
    ('CNRM-CM5', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5',

    ('EC-EARTH', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_ICHEC-EC-EARTH',
    ('EC-EARTH', 'RCA4'): 'SMHI-RCA4_ICHEC-EC-EARTH',

    ('MPI-ESM-LR', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR',
    ('MPI-ESM-LR', 'RCA4'): 'SMHI-RCA4_MPI-M-MPI-ESM-LR',
    ('MPI-ESM-LR', 'REMO2009'): 'MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',

    ('HadGEM2-ES', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
    ('HadGEM2-ES', 'RACMO22E'): 'KNMI-RACMO22E_MOHC-HadGEM2-ES',
    ('HadGEM2-ES', 'RCA4'): 'SMHI-RCA4_MOHC-HadGEM2-ES',

    ('NorESM1-M', 'HIRHAM5'): 'DMI-HIRHAM5_NCC-NorESM1-M',

    ('IPSL-CM5A-MR', 'WRF331F'): 'IPSL-INERIS-WRF331F_IPSL-IPSL-CM5A-MR',
    ('IPSL-CM5A-MR', 'RCA4'): 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
}

_gcm_rcm_couple_adamont_v2_to_full_name = {
    ('CNRM-CM5', 'RACMO22E'): 'KNMI-RACMO22E_CNRM-CERFACS-CNRM-CM5',
    ('CNRM-CM5', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5',
    ('CNRM-CM5', 'RCA4'): 'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5',
    ('CNRM-CM5', 'ALADIN63'): 'CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5',
    ('CNRM-CM5', 'ALADIN53'): 'CNRM-ALADIN53_CNRM-CERFACS-CNRM-CM5',

    ('EC-EARTH', 'RACMO22E'): 'KNMI-RACMO22E_ICHEC-EC-EARTH',
    ('EC-EARTH', 'RCA4'): 'SMHI-RCA4_ICHEC-EC-EARTH',
    ('EC-EARTH', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_ICHEC-EC-EARTH',

    ('MPI-ESM-LR', 'REMO2009'): 'MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
    ('MPI-ESM-LR', 'RCA4'): 'SMHI-RCA4_MPI-M-MPI-ESM-LR',
    ('MPI-ESM-LR', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR',

    ('HadGEM2-ES', 'RACMO22E'): 'KNMI-RACMO22E_MOHC-HadGEM2-ES',
    ('HadGEM2-ES', 'RCA4'): 'SMHI-RCA4_MOHC-HadGEM2-ES',
    ('HadGEM2-ES', 'RegCM4-6'): 'ICTP-RegCM4-6_MOHC-HadGEM2-ES',
    ('HadGEM2-ES', 'CCLM4-8-17'): 'CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',

    ('NorESM1-M', 'REMO2015'): 'GERICS-REMO2015_NCC-NorESM1-M',
    ('NorESM1-M', 'HIRHAM5'): 'DMI-HIRHAM5_NCC-NorESM1-M',

    ('IPSL-CM5A-MR', 'WRF331F'): 'IPSL-INERIS-WRF331F_IPSL-IPSL-CM5A-MR',
    ('IPSL-CM5A-MR', 'RCA4'): 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
    ('IPSL-CM5A-MR', 'WRF381P'): 'IPSL-WRF381P_IPSL-IPSL-CM5A-MR',

    # There was no indicator "max-1day-snowf" for this member
    # For this member there is only the historical anyway
    # ('ERAINT', 'ALADIN62'): 'CNRM-ALADIN62_ECMWF-ERAINT',
}

gcm_rcm_couple_to_color = {
    ('CNRM-CM5', 'CCLM4-8-17'): 'darkred',
    ('CNRM-CM5', 'RCA4'): 'red',
    ('CNRM-CM5', 'ALADIN53'): 'lightcoral',
    # Adamont v2
    ('CNRM-CM5', 'ALADIN63'): 'orangered',
    ('CNRM-CM5', 'RACMO22E'): 'firebrick',

    ('MPI-ESM-LR', 'CCLM4-8-17'): 'darkblue',
    ('MPI-ESM-LR', 'RCA4'): 'blue',
    ('MPI-ESM-LR', 'REMO2009'): 'lightblue',

    ('HadGEM2-ES', 'CCLM4-8-17'): 'darkgreen',
    ('HadGEM2-ES', 'RCA4'): 'green',
    ('HadGEM2-ES', 'RACMO22E'): 'lightgreen',
    # Adamont v2
    ('HadGEM2-ES', 'RegCM4-6'): 'chartreuse',

    ('EC-EARTH', 'CCLM4-8-17'): 'darkviolet',
    ('EC-EARTH', 'RCA4'): 'violet',
    # adamont v2
    ('EC-EARTH', 'RACMO22E'): 'mediumorchid',

    ('IPSL-CM5A-MR', 'WRF331F'): 'darkorange',
    ('IPSL-CM5A-MR', 'RCA4'): 'orange',
    # adamont v2
    ('IPSL-CM5A-MR', 'WRF381P'): 'moccasin',

    ('NorESM1-M', 'HIRHAM5'): 'yellow',
    # adamont v2
    ('NorESM1-M', 'REMO2015'): 'gold',

    # adamont v2
    # ('ERAINT', 'ALADIN62'): 'deeppink'
}

gcm_to_color = {
    'CNRM-CM5': 'red',

    'MPI-ESM-LR': 'tab:blue',

    'HadGEM2-ES': 'green',

    'EC-EARTH': 'violet',

    'IPSL-CM5A-MR': 'orange',

    'NorESM1-M': 'gold',
}

