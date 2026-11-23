"""
Emulator labels, kept free of heavy imports (torch) so that loading the
Gaussian-process emulator does not import the cINN stack.
"""

#: Bundle loaded when no key is given; clients should rely on this default
#: rather than naming the bundle themselves.
DEFAULT_EMULATOR_KEY = "forest_mpg_cinn"
