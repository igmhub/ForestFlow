"""
Priors on the Arinyo and IGM parameters, and the analysis that derives them.

- :mod:`~forestflow.priors.core`: tabulated priors from the DESI DR1 P1D
  posterior (:func:`get_arinyo_priors`, :func:`get_IGM_priors`).
- :mod:`~forestflow.priors.processing`: map P1D posterior samples through the
  emulator and build the sample sets of the analysis.
- :mod:`~forestflow.priors.loaders`: read the external data (BAO) and the
  intermediate products.
- :mod:`~forestflow.priors.importance`: Gaussian fits and importance
  reweighting of the samples.
- :mod:`~forestflow.priors.plots`: figures and tables of the analysis
  (``notebooks/Figures_CM2026``).

Only :mod:`~forestflow.priors.core` is imported with the package; the others
need the optional ``getdist`` (and ``vega``/``astropy`` for the BAO data),
imported only where used. Decoding cup1d P1D chains belongs to cup1d
(``cup1d.postprocessing.forestflow_priors``).
"""
from .core import get_arinyo_priors, get_IGM_priors

__all__ = ["get_arinyo_priors", "get_IGM_priors"]
