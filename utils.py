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


# https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
# does not enable setter, but follow the link to activate setter option

class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self

def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)

if __name__ == '__main__':
    e = Example()
    print(e.big_attribute)
    print(e.big_attribute)
    print(VERSION_TIME)
