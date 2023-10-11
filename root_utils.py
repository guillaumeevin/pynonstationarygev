import datetime
import math
import os.path as op
import subprocess
from itertools import chain
from multiprocessing import Pool

from cached_property import cached_property

VERSION = datetime.datetime.now()
VERSION_TIME = str(VERSION).split('.')[0]
for c in [' ', ':', '-']:
    VERSION_TIME = VERSION_TIME.replace(c, '_')
SHORT_VERSION_TIME = VERSION_TIME[8:]

NB_CORES = 7


def batch_nb_cores(iterable, nb_cores=NB_CORES):
    batchsize = math.ceil(len(iterable) / nb_cores)
    return batch(iterable, batchsize)


def batch(iterable, batchsize=1):
    l = len(iterable)
    for ndx in range(0, l, batchsize):
        yield iterable[ndx:min(ndx + batchsize, l)]


def multiprocessing_batch(function, argument_list, batchsize=None, nb_cores=NB_CORES):
    nb_argument = len(argument_list)
    if batchsize is None:
        batchsize = math.ceil(nb_argument / nb_cores)
    with Pool(nb_cores) as p:
        result_list = p.map(function, batch(argument_list, batchsize=batchsize))
        if None in result_list:
            return None
        else:
            return list(chain.from_iterable(result_list))


def terminal_command(command_str):
    return subprocess.check_output(command_str.split()).decode("utf-8").split('\n')


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

def memoize(function):
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv

    return wrapper

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
