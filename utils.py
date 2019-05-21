import datetime
import os.path as op
from cached_property import cached_property

VERSION = datetime.datetime.now()
VERSION_TIME = str(VERSION).split('.')[0]
for c in [' ', ':', '-']:
    VERSION_TIME = VERSION_TIME.replace(c, '_')

NB_CORES = 7

def get_root_path() -> str:
    return op.dirname(op.abspath(__file__))


def get_full_path(relative_path: str) -> str:
    return op.join(get_root_path(), relative_path)


def get_display_name_from_object_type(object_type):
    # assert isinstance(object_type, type), object_type
    return str(object_type).split('.')[-1].split("'")[0].split(' ')[0]


def first(s):
    """Return the first element from an ordered collection
       or an arbitrary element from an unordered collection.
       Raise StopIteration if the collection is empty.
    """
    return next(iter(s))


def float_to_str_with_only_some_significant_digits(f, nb_digits) -> str:
    assert isinstance(nb_digits, int)
    assert nb_digits > 0
    return '%s' % float('%.{}g'.format(nb_digits) % f)


class Example(object):

    @cached_property
    def big_attribute(self):
        print('Long loading only once...')
        return 2


if __name__ == '__main__':
    e = Example()
    print(e.big_attribute)
    print(e.big_attribute)
    print(VERSION_TIME)
