import os.path as op


def get_root_path() -> str:
    return op.dirname(op.abspath(__file__))


def get_full_path(relative_path: str) -> str:
    return op.join(get_root_path(), relative_path)


def get_display_name_from_object_type(object_type):
    # assert isinstance(object_type, type), object_type
    return str(object_type).split('.')[-1].split("'")[0]
