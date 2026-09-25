Scientific conventions and contracts
====================================

Units and names
---------------

Dimensional public names include a unit suffix. The letter ``i`` means
inverse: ``k_iMpc`` is in Mpc^-1 and ``k_ikms`` is in
(km/s)^-1. Power spectra use capitalized names: ``P1D_Mpc`` is in Mpc,
``P3D_Mpc`` is in Mpc^3, and ``P1D_kms`` is in km/s.
The conversion ``dkms_diMpc = H(z)/(1+z)`` is in (km/s)/Mpc.

Thus ``k_iMpc = k_ikms * dkms_diMpc`` and
``P1D_kms = P1D_Mpc * dkms_diMpc``. Conversion of P3D uses the
third power. Dimensionless quantities such as ``z`` and ``mu`` have no
unit suffix.

Array contracts
---------------

Public wavenumber arrays are finite and positive. For ``P3D_Mpc``, the
wavenumber and ``mu`` inputs must be broadcast-compatible and the output has
their broadcast shape. For ``P1D_Mpc`` and ``P1D_kms``, a one-dimensional
input of length ``Nk`` produces shape ``(Nk,)``. Batched redshift/model
axes precede the wavenumber axes. Covariances have shape ``(N, N)``;
standard deviations have the same shape and units as the corresponding power.
Uncertainty outputs must state whether they are standard deviations,
variances, covariances, or fractional errors.

Parameter ordering and validity
-------------------------------

Ordered Arinyo arrays follow ``ARINYO_PARAMETER_NAMES`` from
``forestflow.conventions``. Mappings are preferred at public boundaries.
An emulator's stored ``input_labels`` and normalization metadata define its
input order and training domain. Calls outside that domain are extrapolations
and must not be interpreted as validated predictions.

Legacy data
-----------

Archives with ``k_Mpc``, ``p1d_Mpc``, ``p3d_Mpc``,
``k_kms``, ``Pk_kms``, or ``dkms_dMpc`` remain readable through
compatibility normalization. New interfaces and stored products should use
the canonical names above.
