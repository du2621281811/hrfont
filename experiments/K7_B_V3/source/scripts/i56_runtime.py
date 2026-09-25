"""I5=main-only, I6=+Chinese auxiliary. Same I0 initialization as I3/I4."""
from i34_runtime import *
from i34_runtime import model_for as _model_for_i34


def model_for(arm, device, output):
    if arm not in ('I5', 'I6'):
        raise ValueError(arm)
    model, args = _model_for_i34('I3' if arm == 'I5' else 'I4', device, output)
    model.arm = arm
    return model, args
