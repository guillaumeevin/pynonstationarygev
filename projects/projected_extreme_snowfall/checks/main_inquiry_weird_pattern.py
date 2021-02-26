from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies

def main_rcm():
    for rcm in ['CCLM4-8-17', 'RACMO22E', 'RCA4', 'ALADIN63', 'ALADIN53']:
        altitude_studies = AltitudesStudies(AdamontSnowfall, altitudes=[600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600],
                                            scenario=AdamontScenario.rcp85, gcm_rcm_couple=('CNRM-CM5', rcm))
        altitude_studies.plot_maxima_time_series()

def main_gcm():
    for gcm in ['CNRM-CM5', 'EC-EARTH', 'HadGEM2-ES']:
        altitude_studies = AltitudesStudies(AdamontSnowfall,
                                            altitudes=[600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600],
                                            scenario=AdamontScenario.rcp85, gcm_rcm_couple=(gcm, 'CCLM4-8-17'))
        altitude_studies.plot_maxima_time_series()

if __name__ == '__main__':
    main_gcm()
    # main_rcm()
    
    
    