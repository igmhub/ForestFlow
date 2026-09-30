"""Array-shape helpers shared across ForestFlow models."""

import numpy as np


def broadcast_leading_dimensions(values, target):
    """Add trailing singleton dimensions so ``values`` broadcasts over ``target``."""
    values = np.asarray(values)
    return values.reshape(values.shape + (1,) * (np.ndim(target) - values.ndim))
