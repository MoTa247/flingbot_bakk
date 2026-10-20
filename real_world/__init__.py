import os as _os
from importlib import import_module as _import_module

# GEMSORT: importing the UR5 stack opens sockets to the UR5 grippers at import time (UR5Pair's default argument calls
# setup.get_ur5s()), which fails on a cell that has no UR5s. With GEMSORT_ROBOTS=1 (default here) the UR5 classes are
# only placeholders; set GEMSORT_ROBOTS=0 to import the original hardware stack.
if _os.environ.get('GEMSORT_ROBOTS', '1') == '1':
    class UR5MoveTimeoutException(Exception):
        """Placeholder: the real one lives in realur5.py and is only imported with the UR5 backend."""

    class _UR5Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('UR5 backend not imported (GEMSORT_ROBOTS=1). Use GemsortArmPair, or set '
                               'GEMSORT_ROBOTS=0 to talk to real UR5s.')

    UR5 = UR5Pair = _UR5Unavailable
else:
    from .realur5 import UR5, UR5MoveTimeoutException
    from .ur5_pair import UR5Pair

# Keep the package itself lightweight. Python executes this file before every
# ``python -m real_world.<tool>`` command; eagerly importing fling/stretch used
# to pull in the learning package and Ray even for camera calibration. Resolve
# the original public API only when a policy/runtime caller actually requests
# one of these names.
_LAZY_EXPORTS = {
    'WSG50': ('.wsg50', 'WSG50'),
    'RG2': ('.rg2', 'RG2'),
    'fling': ('.fling', 'fling'),
    'stretch': ('.stretch', 'stretch'),
    'pick_and_drop': ('.reset_cloth', 'pick_and_drop'),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exc
    value = getattr(_import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = ['UR5', 'UR5Pair', 'WSG50', 'RG2',
           'UR5MoveTimeoutException',
           'stretch', 'fling', 'pick_and_drop']
