from typing import Dict, List

from experiment.eurocode_data.departement_alpes_francaises import HauteSavoie, Savoie, Isere, Drome, HautesAlpes, \
    AlpesDeHauteProvence, AlpesMaritimes, AbstractDepartementAlpesFrancaises

massif_name_to_departement_types = {
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

massif_name_to_departement_objects = {m: [d() for d in deps] for m, deps in
                                      massif_name_to_departement_types.items()}  # type: Dict[str, List[AbstractDepartementAlpesFrancaises]]

DEPARTEMENT_TYPES = [HauteSavoie, Savoie, Isere, Drome, HautesAlpes, AlpesMaritimes, AlpesDeHauteProvence]

dep_class_to_massif_names = {dep: [k for k, v in massif_name_to_departement_types.items() if dep in v]
                             for dep in DEPARTEMENT_TYPES
                             }

if __name__ == '__main__':
    for k, v in dep_class_to_massif_names.items():
        print(k, v)
