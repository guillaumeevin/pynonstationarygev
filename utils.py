import os
import os.path as op


def get_root_path() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_full_path(relative_path: str) -> str:
    return op.join(get_root_path(), relative_path)

