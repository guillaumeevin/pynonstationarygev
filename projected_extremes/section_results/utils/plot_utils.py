
def add_suffix_label(label, massif_name, relative_change):
    if massif_name is not None:
        label += ' for the {} massif with\nrespect to'.format(massif_name.replace('_', '-'))
    else:
        label += ' with respect to'
    unit = '(\%)' if relative_change else '(kg m$^{-2}$)'
    label += ' + 1$^o\\textrm{C}$ of\nglobal warming above pre-industrial levels ' + unit
    return label
