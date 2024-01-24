# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: lace
# ---

# # Generate GRF for testing

import sys
sys.path.append("../scripts")
sys.path.append("../forestflow")
import Px_functions
import pcross
import numpy as np
import matplotlib.pyplot as plt
# get predictions from P-cross integral
from lace.cosmo import camb_cosmo
from ForestFlow.model_p3d_arinyo import ArinyoModel


# +
# make a power spectrum
def P(k):
    # small at low k
    taper = np.exp(-.1 * (k - 10))

    taper[k<10] = 1
    return k * np.exp(-2*k) + np.exp(-2/k)*k**-3 * taper
    # return 100*(1 / (k+.1)**2)
    # return (k+2)**.5
    

k3ds = np.logspace(-2,2,100)
plt.plot(k3ds, P(k3ds))
plt.ylabel("P(k) [Mpc]")
plt.xlabel("k [1/Mpc]")
# plt.plot(k3ds, k3ds)
# plt.plot(k3ds, k3ds**-3)
plt.yscale('log')
plt.xscale('log')
plt.ylim([10**-6,10])
# add a cutoff at high k (gas pressure)
# make power more realistic (propto k at low k)
# -

# For testing, we want our pixel size to be the same as the actual skewers but to smooth the field slightly larger than that
#

max_k = 20 # actually 60 Mpc^-1
pix_size = 2*np.pi/max_k # Mpc
# pix_size = 0.05 # Mpc, same as in sims
print(pix_size)

# +
L = 67.5 # Mpc
npix = int(L / pix_size)
print(npix)
# real_field = Px_functions.generate_grf(L, P, npix_per_side=npix)

import density_field_library as DFL

grid              = npix    #grid size
BoxSize           = L*0.7 #Mpc/h
seed              = 1      #value of the initial random seed
Rayleigh_sampling = 0      #whether sampling the Rayleigh distribution for modes amplitudes
threads           = 1      #number of openmp threads
verbose           = True   #whether to print some information

# read power spectrum; k and Pk have to be floats, not doubles
k, Pk = k3ds, P(k3ds)
k, Pk = k.astype(np.float32), Pk.astype(np.float32)


# generate a 3D Gaussian density field
df_3D = DFL.gaussian_field_3D(grid, k, Pk, Rayleigh_sampling, seed,
                              BoxSize, threads, verbose)
# -

plt.imshow(df_3D[0,:,:])
plt.colorbar()

# +
# # vs my way
# real_field = Px_functions.generate_grf(L, P, npix_per_side=npix)
# plt.imshow(real_field[0,:,:])
# plt.colorbar()
# # mine is normalized weirdly I guess 
# -

# Make sure we can recover the same power from this GRF, in the right units

# +
import Pk_library as PKL

MAS     = 'CIC'  #mass-assigment scheme

Pk = PKL.Pk(df_3D, BoxSize, 0, MAS, threads, verbose)

# 3D P(k)
k       = Pk.k3D
Pk0     = Pk.Pk[:,0] #monopole
Pk2     = Pk.Pk[:,1] #quadrupole
Pk4     = Pk.Pk[:,2] #hexadecapole
Pkphase = Pk.Pkphase #power spectrum of the phases
Nmodes  = Pk.Nmodes3D

plt.loglog(k3ds, P(k3ds), label='input')
plt.loglog(k, Pk0, label='measured')
plt.ylim([10**-6,10])
plt.legend()
plt.ylabel("P(k) [Mpc]")
plt.xlabel("k [1/Mpc]")


# +
# def calculate_power_spectrum(gaussian_field, pixel_size):
#     # Perform 3D Fourier transform
#     fourier_transform = np.fft.fftn(gaussian_field)

#     # Calculate the squared magnitude of the Fourier coefficients
#     power_spectrum = np.abs(fourier_transform)**2

#     # Calculate the spherically averaged power spectrum
#     field_size = gaussian_field.shape[0]
#     k_values = np.fft.fftfreq(field_size, d=pixel_size)  # Frequency values
#     k = np.sqrt(np.sum(np.array(np.meshgrid(k_values, k_values, k_values, indexing='ij'))**2, axis=0))  # Radial frequency


#     # Bin the power spectrum values based on radial frequency
#     num_bins = int(field_size // 2)  # Number of bins
#     hist, bin_edges = np.histogram(k, bins=num_bins, weights=power_spectrum)

#     # Calculate the average power in each bin
#     avg_power_spectrum = hist / np.diff(bin_edges)

#     return bin_edges[1:], avg_power_spectrum[1:]  # Exclude the zero-frequency bin
# bin_edges, avg_power = calculate_power_spectrum(df_3D, pix_size)
# plt.plot(bin_edges[1:], avg_power/L**3)

# +
# smooth the GRF
from scipy.ndimage import gaussian_filter
# smoothing should be similar to pixel size in real skewers, .09 Mpc

# smoothing_sigma = 0.09 / pix_size
smoothing_sigma=1
print(smoothing_sigma)
smoothed_field = gaussian_filter(df_3D, sigma=smoothing_sigma)


# +
# def gaussian(k, mean, std_dev):
#     return np.exp(-0.5 * ((k - mean) / std_dev)**2)
# ks = np.linspace(0,13,100)
# plt.plot(ks, gaussian(ks, 10, smoothing_sigma))
# plt.xlim([0,17.5])
# -

plt.show()
plt.imshow(smoothed_field[0,:,:])

Pksmooth = PKL.Pk(smoothed_field, BoxSize, 0, MAS, threads, verbose)
# 3D P(k)
ksmooth       = Pksmooth.k3D
Pk0smooth     = Pksmooth.Pk[:,0] #monopole


plt.plot(k3ds, P(k3ds), label='P(k) input')
plt.plot(k, Pk0, label='P(k) measured (no smooth)')
plt.plot(ksmooth, Pk0smooth, label='P(k) measured (smooth)')
plt.yscale('log')
plt.xscale('log')
plt.ylim([10**-6,10])
plt.ylabel("P(k) [Mpc]")
plt.xlabel("k [1/Mpc]")
plt.legend()
plt.axvline(.1, color='k')
plt.axvline(2, color='k')

# +
# # make a 3D plot
# # %matplotlib inline
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax  = fig.add_subplot(111,projection='3d')
# shape = real_field.shape
# x   = np.arange(0, shape[0], 1)
# y=x
# z=x
# x,y,z = np.meshgrid(x,y,z)

# ax.scatter(x, y, z, c=smoothed_field, alpha=0.5, cmap='viridis', edgecolor=None, s=35)
# plt.title(r"$P(k)\propto k^2$")
# # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
# # cbar.set_label('Field Values')
# # ax.set_xlabel('X')
# # ax.set_ylabel('Y')
# # ax.set_zlabel('Z')

# # # Set title for the plot
# # ax.set_title('3D Gaussian Field')

# # plt.show()

# +
nskew_per_side = 40
ix = np.arange(int(npix//nskew_per_side), smoothed_field.shape[0], int(npix//nskew_per_side))
iy = np.arange(int(npix//nskew_per_side), smoothed_field.shape[0], int(npix//nskew_per_side))

points    = np.array([(x, y) for x in ix for y in iy])
positions = points * L / smoothed_field.shape[0]
# -

plt.imshow(smoothed_field[:,:,1])
plt.scatter(points[:,0], points[:,1], color='k', s=2)

skewers_ax1 = smoothed_field[points[:,0], points[:,1], :]

for skewer in skewers_ax1[:5]:
    plt.plot(skewer)

# +
Ns,Np = skewers_ax1.shape

print(Ns, Np)
size = int(np.sqrt(Ns))
position_grid  = positions.reshape((size,size,2))
skewers_ax1 = skewers_ax1.reshape((size,size,Np))
delta_flux_k = np.fft.rfft(skewers_ax1) # Fourier transform all the skewers

# +
dz = np.linalg.norm(position_grid[0,1]-position_grid[0,0])
print(dz) # actual Mpc separation between positions

spacing_dL = np.array([1,2,4,6]) # the integer number of dz that corresponds to that spacing
print(spacing_dL)
spacing = spacing_dL*dz
print(spacing) # the actual spacings between skewers

# +
Px = []
Px_errs = []

for dL in spacing_dL:
    Px_dL = []
    Px_dL_errs = []
    print(dL)
    # i, j = np.meshgrid(np.arange(size-dL), np.arange(size-dL), indexing='ij') # faster but takes too much memory
    for i in range(size-dL):
        for j in range(size-dL):
            # vectorized version
            # Px_phase1=np.real(delta_flux_k[phase][i,j]*np.conjugate(delta_flux_k[phase][i,j+dL])) # 'loop' through 0 axis
            # Px_phase2=np.real(delta_flux_k[phase][i,j]*np.conjugate(delta_flux_k[phase][i+dL,j])) # loop through 1 axis
            # Px_phase = np.concatenate((Px_phase1,Px_phase2))
            # del Px_phase1
            # del Px_phase2
            Px_dL.append(np.real(delta_flux_k[i,j]*np.conjugate(delta_flux_k[i,j+dL]))) # loop through 0 axis
            if dL!=0:
                Px_dL.append(np.real(delta_flux_k[i,j]*np.conjugate(delta_flux_k[i+dL,j]))) # loop through 1 axis
            # need to account here for periodic boundaries
            if i==20 and j==20:
                print('position diff', np.linalg.norm(position_grid[i,j] - position_grid[i,j+dL]))
                print(np.linalg.norm(position_grid[i,j] - position_grid[i+dL,j]))


    # after looping through both phases, average all the results
    Px_dL = np.asarray(Px_dL)
    avg_over_phases = np.average(Px_dL, axis=0)*(L/(Np**2))
    print(avg_over_phases.shape)
    Px.append(avg_over_phases)
    print(len(Px_dL))
    errs = np.std(Px_dL, axis=0)*(L/(Np**2))/np.sqrt(len(Px_dL))
    Px_errs.append(errs)
# -

pix_spacing = L/Np
kpar = np.fft.rfftfreq(Np, pix_spacing)*2*np.pi # frequency in Mpc^-1

# +
# now get the corresponding Pcross predictions

import numpy as np


def get_Px(
    kpars,
    P3D,
    fast=False,
    min_rperp=0.01,
    max_rperp=30,
    min_kperp=10.0**-20,
    max_kperp=10.0**3,
    Nsteps_kperp=5000,
    trans_to_p1d=True,
    fast_transition=False
):
    """ Calculates P_cross, the power for a given k_parallel mode from pairs of lines-of-sight separated by perpendicular distance rperp, given a 3D power spectrum
    Calculation is done with the hankl transform.

    Required Parameters:
        kpars (array): array of k parallel (usually log-spaced)    
        params (dictionary): parameters for the Arinyo parameters. If not given, they will be set to default values
        z (float): single redshift to evaluate
    Optional Parameters:
        fast (bool): if true, accuracy is <0.1% and the runtime is ~0.8s for 100 kpar values and all other default settings. If false, accuracy is <1% and the runtime is ~0.25s
        min_rperp, max_rperp (float): desired range of rperp values to return
        min_kperp, max_kperp (float): range of kperp values to use in the calculation. Decreasing this range can cause unwanted artifacts
        Nsteps_kperp (int): number of kperps for the hankl transform (and number of output rperp). Decreasing this speeds up calculation but decreases accuracy
        trans_to_p1d (bool): determines whether to transition to the P1D result at low rperp
        fast_transition (bool): if true, the transition to P1D is done faster, without interpolation, but there will be a discontinuity

    Returns:
        rperp, Px_per_rperp, Px_per_kpar
        rperp: array of log-space r-perpendicular (separation in Mpc)
        Px_per_kpar: P-cross as an array with shape (len(kpars), len(rperp)).
    """
    import hankl
    from scipy.interpolate import CubicSpline

    if fast:
        Nsteps_kperp = 1000
    if min_rperp > 0.08:
        trans_to_p1d = False  # not necessary to transition to the P1D result if minimum requested rperp is larger than 0.08

    if min_rperp > 0.2 and fast:
        Nsteps_kperp = 500  # speed it up because we will cut out the range of low-rperp oscillations

    Px_per_kpar = []
    for kpar in kpars:  # for each value of k parallel to evaluate Px at
        kperps = np.logspace(
            np.log10(min_kperp), np.log10(max_kperp), Nsteps_kperp
        )  # set up an array of kperp
        kpars_prime = np.full(
            len(kperps), kpar
        )  # each kperp gets the same kpar for this iteration -- make a full array of the kpar value
        k = np.sqrt(kpars_prime**2 + kperps**2)  # get the corresponding k array
        func = P3D(k) * 0.5 * np.sqrt(kperps / (2 * np.pi))
        rperp, LHS = hankl.FFTLog(
            kperps, func, q=0, mu=0.5
        )  # returns an array of log-spaced rperps, and the Hankel Transform
        Px = LHS / (rperp ** (3 / 2))  # Divide out by remaining factor to get Px
        if min_rperp > min(rperp):
            rperp_minidx = np.argmin(abs(rperp - min_rperp))
        if max_rperp < max(rperp):
            rperp_maxidx = np.argmin(abs(rperp - max_rperp))
        else:
            rperp_minidx, rperp_maxidx = None, None
        Px_per_kpar.append(Px[rperp_minidx:rperp_maxidx])
        rperp = rperp[rperp_minidx:rperp_maxidx]
    Px_per_kpar = np.asarray(Px_per_kpar)

    return rperp, Px_per_kpar



# -

rperp,Px_per_kpar = get_Px(kpar, P)


# find the closest predicted rperps to the measured skewer spacings
idxs = []
for i in range(len(spacing)):
    idxs.append(np.argmin(abs(rperp-spacing[i])))
    print(rperp[idxs[i]])

#min frequency, max frequency
Np = df_3D.shape[0]
print(f"We can't expect to measure anything smaller than k={2*np.pi/L} or larger than k={np.pi/(L/Np)}")

cmap[0]


import matplotlib as mpl
cmap = mpl.colormaps['Set1']
for i in range(len(spacing)):
    plt.plot(kpar, Px[i], label=f'spacing = {round(spacing[i],1)}, skewers', color=cmap(i/len(spacing)), linestyle='dashed')
    plt.plot(kpar, Px_per_kpar.T[idxs[i]], label=f'spacing = {round(spacing[i],1)}, integral', color=cmap(i/len(spacing)))
# plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.xlim([0.1,2])


