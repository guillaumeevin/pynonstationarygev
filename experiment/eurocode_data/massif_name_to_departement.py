from typing import Dict, List

from experiment.eurocode_data.departementalpesfrancaises import HauteSavoie, Savoie, Isere, Drome, HautesAlpes, \
    AlpesDeHauteProvence, AlpesMaritimes, AbstractDepartementAlpesFrancaises

massif_name_to_departements = {
    'Chablais': [HauteSavoie],
    'Aravis': [HauteSavoie, Savoie],
    'Mont-Blanc': [HauteSavoie],
    'Bauges': [HauteSavoie, Savoie],
    'Beaufortain': [HauteSavoie, Savoie],
    'Haute-Tarentaise': [Savoie],
    'Chartreuse': [Isere, Savoie],
    'Belledonne': [Isere, Savoie],
    'Maurienne': [Savoie],
    'Vanoise': [Savoie],
    'Haute-Maurienne': [Savoie],
    'Grandes-Rousses': [Isere, Savoie],
    'Thabor': [HauteSavoie],
    'Vercors': [Isere, Drome],
    'Oisans': [Isere, HautesAlpes],
    'Pelvoux': [Isere, HautesAlpes],
    'Queyras': [HautesAlpes],
    'Devoluy': [Drome, Isere, HautesAlpes],
    'Champsaur': [HautesAlpes],
    'Parpaillon': [HautesAlpes, AlpesDeHauteProvence],
    'Ubaye': [AlpesDeHauteProvence],
    'Haut_Var-Haut_Verdon': [AlpesDeHauteProvence],
    'Mercantour': [AlpesMaritimes, AlpesDeHauteProvence]}
massif_name_to_departements = {m: [d() for d in deps] for m, deps in massif_name_to_departements.items()}  # type: Dict[str, List[AbstractDepartementAlpesFrancaises]]
