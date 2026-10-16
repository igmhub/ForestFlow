Backup (2026-10-16) of the emulator-dependent cached products before switching the
Figures_CM2026 pipeline to `forest_mpg_gp`.

These were produced with the legacy cINN (`data/emulator_models/mpg_hypercube.pt`,
`old_code/old_emu/paper_P3D_cINN`, training_type="Arinyo_min"), i.e.
`set_map_igm_p3d(..., old_emu=True)`, on 2026-06-04.

- arinyo_from_{desi,accel2}_p1d.npy: P1D chain -> Arinyo mapping (+P1D/P3D for accel2)
- priors_{desi,accel2}_priors_arinyo_from_p1d.npy: copies of ForestFlow/data/priors_*

The chain-derived inputs (*_priors_cosmo_IGM_from_p1d.npy, *_inter_chain.npy, sig8_*)
do not depend on the emulator and were left in place in int_data_figs/.
