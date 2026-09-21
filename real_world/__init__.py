import os as _os

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
from .wsg50 import WSG50
from .rg2 import RG2
from .fling import fling
from .stretch import stretch
from .reset_cloth import pick_and_drop

__all__ = ['UR5', 'UR5Pair', 'WSG50', 'RG2',
           'UR5MoveTimeoutException',
           'stretch', 'fling', 'pick_and_drop']
