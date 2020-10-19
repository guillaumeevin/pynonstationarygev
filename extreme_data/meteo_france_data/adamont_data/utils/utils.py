from enum import Enum


def _massif_number_to_massif_name():
    # from adamont_data metadata
    s = """1	Chablais
    2	Aravis
    3	Mont-Blanc
    4	Bauges
    5	Beaufortain
    6	Haute-Tarentaise
    7	Chartreuse
    8	Belledonne
    9	Maurienne
    10	Vanoise
    11	Haute-Maurienne
    12	Grandes-Rousses
    13	Thabor
    14	Vercors
    15	Oisans
    16	Pelvoux
    17	Queyras
    18	Devoluy
    19	Champsaur
    20	Parpaillon
    21	Ubaye
    22	Haut_Var-Haut_Verdon
    23	Mercantour"""
    l = s.split('\n')
    return {int(k): m for k, m in dict([e.split() for e in l]).items()}


massif_number_to_massif_name = _massif_number_to_massif_name()
