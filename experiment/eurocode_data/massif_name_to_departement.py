from typing import Dict, List

from experiment.eurocode_data.departement_alpes_francaises import HauteSavoie, Savoie, Isere, Drome, HautesAlpes, \
    AlpesDeHauteProvence, AlpesMaritimes, AbstractDepartementAlpesFrancaises
from experiment.eurocode_data.eurocode_region import C1, C2, E

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
    'Mercantour': [AlpesMaritimes, AlpesDeHauteProvence]
}

massif_name_to_eurocode_region = {
    'Chablais': E,
    'Aravis': E,
    'Mont-Blanc': E,
    'Bauges': E,
    'Beaufortain': E,
    'Haute-Tarentaise': E,
    'Chartreuse': C2, # Mainly in Isère (and small part in Savoie but belong to C2 cantons)
    'Belledonne': C2, # Mainly in Isère (and small part in Savoie but belong to C2 cantons)
    'Maurienne': E,
    'Vanoise': E,
    'Haute-Maurienne': E,
    'Grandes-Rousses': E, # Saint-Sorlin-d'Arves belong to the Grandes ROusses, and belong to the canton of St Jean de Maurienne in Savoie which is considered E
    'Thabor': E,
    'Vercors': C2,
    'Oisans': C2, # we consider they are mainly in Isere, thus C2
    'Pelvoux': C2,
    'Queyras': C1,
    'Devoluy': C1, # Look on Google Map, but when we look at the mountain ranges devoluy is clearly only in Hautes Alpes
    'Champsaur': C1,
    'Parpaillon': C1,
    'Ubaye': C1,
    'Haut_Var-Haut_Verdon': C1,
    'Mercantour': C1
}

massif_name_to_departement_objects = {m: [d() for d in deps] for m, deps in
                                      massif_name_to_departement_types.items()}  # type: Dict[str, List[AbstractDepartementAlpesFrancaises]]

DEPARTEMENT_TYPES = [HauteSavoie, Savoie, Isere, Drome, HautesAlpes, AlpesMaritimes, AlpesDeHauteProvence]

dep_class_to_massif_names = {dep: [k for k, v in massif_name_to_departement_types.items() if dep in v]
                             for dep in DEPARTEMENT_TYPES
                             }

if __name__ == '__main__':
    for k, v in dep_class_to_massif_names.items():
        print(k, v)
