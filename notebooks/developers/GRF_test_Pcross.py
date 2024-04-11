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

# +
import sys
import numpy as np
import matplotlib.pyplot as plt
# get predictions from P-cross integral
from lace.cosmo import camb_cosmo
# %load_ext autoreload
# %autoreload 2

from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.pcross import get_Px
import matplotlib as mpl

cmap = mpl.colormaps['Set1']


# -

# We want to generate a power spectrum that rises and falls like the matter power spectrum
# It should span the same k range as the sims, ideally
# The sims have length ~67 Mpc so lowest k ~ 0.1
# The sims have pixel spacing 0.05 Mpc, so highest k = pi / 0.05 ~ 60 Mpc-1

# +
# smooth = 2 # default
# smooth = 5
# make a power spectrum
def P(z, kpar, kperp, smooth=False, smoothing=2):
    # small at low k
    k = kpar**2 + kperp**2
    taper = np.exp(-.1 * ((k-10)**2))
    if len(k)>1:
        taper[k<10] = 1
    elif len(k)<=1:
        if k<10:
            taper = 1
    if smooth:
        # return k * np.exp(-2*k) + np.exp(-2/k)*k**-3 * np.exp(-k**2*2)
        return (k * np.exp(-2*k) + np.exp(-2/k)*k**-3) * np.exp(-k**2*smoothing)
    else:
        return (k * np.exp(-2*k) + np.exp(-2/k)*k**-3) * taper
    # return np.full(k.shape,5) # white noise

def P_iso(z, k, smooth=False, smoothing=2):
    # small at low k
    taper = np.exp(-.1 * ((k-10)**2))
    if len(k)>1:
        taper[k<10] = 1
    elif len(k)<=1:
        if k<10:
            taper = 1
    if smooth:
        # return k * np.exp(-2*k) + np.exp(-2/k)*k**-3 * np.exp(-k**2*2)
        return (k * np.exp(-2*k) + np.exp(-2/k)*k**-3) * np.exp(-k**2*smoothing)
    else:
        return (k * np.exp(-2*k) + np.exp(-2/k)*k**-3) * taper
    # return np.full(k.shape,5) # white noise
    
def P_pol(z, k, mu, smooth=False, smoothing=2):
    # small at low k
    taper = np.exp(-.1 * ((k-10)**2))
    if len(k)>1:
        taper[k<10] = 1
    elif len(k)<=1:
        if k<10:
            taper = 1
    if smooth:
        # return k * np.exp(-2*k) + np.exp(-2/k)*k**-3 * np.exp(-k**2*2)
        return (k * np.exp(-2*k) + np.exp(-2/k)*k**-3) * np.exp(-k**2*smoothing)
    else:
        return (k * np.exp(-2*k) + np.exp(-2/k)*k**-3) * taper
    # return np.full(k.shape,5) # white noise

nn_k3d = 500
kpar = np.logspace(-3, np.log10(10), nn_k3d)
kperp = np.logspace(-3, np.log10(10), nn_k3d)
kperp2d  = np.tile(kperp[:, np.newaxis], len(kpar)) # mu grid for P3D
kpar2d   = np.tile(kpar[:, np.newaxis], len(kperp)).T
k3ds = kperp2d**2+kpar2d**2
ps = P(0, kpar2d, kperp2d)
kparchoice = 40
plt.plot(k3ds[:,kparchoice], ps[:,kparchoice], label='original')
smoothing=.1
plt.plot(k3ds[:,kparchoice], P(0, kpar2d, kperp2d, smooth=True, smoothing=smoothing)[:,kparchoice],label='smoothed')
plt.plot(k3ds[:,kparchoice], np.exp(-k3ds**2*smoothing)[:,kparchoice], label='smoothing kernel')
k3d_iso = np.logspace(-2, np.log10(100), nn_k3d)
# plt.plot(k3d_iso, P_iso(0, k3d_iso, smooth=True, smoothing=smoothing), label='iso', linestyle='dashed')
plt.ylabel("P(k) [Mpc]")
plt.xlabel(r"$k$ [Mpc$^{-1}$]")
# plt.plot(k3ds, k3ds)
# plt.plot(k3ds, k3ds**-3)
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.ylim([10**-10,10])
plt.xlim([10**-2,20])
# add a cutoff at high k (gas pressure)
# make power more realistic (propto k at low k)

plt.axvline(2*np.pi/L, color='k')
plt.axvline(np.pi/pix_size/2, color='k')
# -

# For testing, we want our pixel size to be the same as the actual skewers but to smooth the field slightly larger than that
#

# +
pix_size = 0.1 # Mpc
L = 100 # Mpc
# npix_perside = 2**7
npix_perside = L/pix_size
pix_size = L/npix_perside
print(npix_perside, pix_size)
# real_field = Px_functions.generate_grf(L, P, npix_per_side=npix)
max_k = np.pi/pix_size # actually 60 Mpc^-1
# pix_size = np.pi/max_k # Mpc
# pix_size = 0.05 # Mpc, same as in sims
print(max_k)

import density_field_library as DFL

grid              = npix_perside    #grid size
BoxSize           = L #Mpc
seed              = 1      #value of the initial random seed
Rayleigh_sampling = 0      #whether sampling the Rayleigh distribution for modes amplitudes
threads           = 1      #number of openmp threads
verbose           = True   #whether to print some information

# read power spectrum; k and Pk have to be floats, not doubles
k, Pk = k3ds[:,kparchoice], P(0, kpar2d, kperp2d, smooth=True, smoothing=smoothing)[:,kparchoice]
k, Pk = k.astype(np.float32), Pk.astype(np.float32)

# generate a 3D Gaussian density field
df_3D = DFL.gaussian_field_3D(grid, k, Pk, Rayleigh_sampling, seed,
                              BoxSize, threads, verbose)

plt.imshow(df_3D[1,:,:])
plt.colorbar()

# # vs my way
# real_field = Px_functions.generate_grf(L, P, npix_per_side=npix_perside)
# plt.imshow(real_field[0,:,:])
# plt.colorbar()
# # mine is normalized weirdly I guess 
# -

# Make sure we can recover the same power from this GRF, in the right units

# +
# import Pk_library as PKL

# MAS     = 'CIC'  #mass-assigment scheme

# Pk = PKL.Pk(df_3D, BoxSize, 0, MAS, threads, verbose)

# # 3D P(k)
# k       = Pk.k3D
# Pk0     = Pk.Pk[:,0] #monopole
# Pk2     = Pk.Pk[:,1] #quadrupole
# Pk4     = Pk.Pk[:,2] #hexadecapole
# Pkphase = Pk.Pkphase #power spectrum of the phases
# Nmodes  = Pk.Nmodes3D

# plt.loglog(k3ds[:,kparchoice], P(0, kpar2d, kperp2d, smooth=True, smoothing=smoothing)[:,kparchoice], label='input')
# plt.loglog(k, Pk0, label='measured')
# plt.ylim([10**-10,10])
# plt.legend()
# plt.ylabel("P(k) [Mpc]")
# plt.xlabel("k [1/Mpc]")
# plt.axvline(2*np.pi/L, color='k')
# plt.axvline(np.pi/pix_size/2, color='k')

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
# # smooth the GRF
# from scipy.ndimage import gaussian_filter
# # smoothing should be similar to pixel size in real skewers, .09 Mpc

# # smoothing_sigma = 0.09 / pix_size
# smoothing_sigma=10
# print(smoothing_sigma) # in pixels
# smoothed_field = gaussian_filter(df_3D, sigma=smoothing_sigma)

# def gaussian_real(x, mean, sigma):
#     return np.exp(-0.5 * ((x - mean) / sigma)**2)
# def gaussian_fourier(k, mean, sigma):
#     return (np.pi*np.sqrt(2)*sigma)**(3/2)*np.exp(-k**2*np.sqrt(2)*sigma/4)

# xarr = np.linspace(0,2,100)
# plt.plot(xarr, gaussian_real(xarr, 0, smoothing_sigma*pix_size), label="Gaussian filter in real")
# plt.xlabel("x or k")
# plt.plot(xarr, gaussian_fourier(xarr, 0, smoothing_sigma*pix_size), label="Gaussian filter in Fourier")
# plt.title("Gaussian filter in real v fourier")
# plt.legend()



# +
# import smoothing_library as SL

# R       = 10*pix_size #Mpc
# grid    = df_3D.shape[0]
# Filter  = 'Gaussian'
# threads = 1

# # compute FFT of the filter
# W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)

# # smooth the field
# smoothed_field = SL.field_smoothing(df_3D, W_k, threads)
# plt.show()
# plt.imshow(smoothed_field[0,:,:])

# +
# xarr = np.linspace(0,5,10)
# plt.plot(xarr, gaussian_real(xarr, 0, R), label="Gaussian filter in real")
# plt.xlabel("x or k")
# plt.plot(np.insert(Pk.k1D, 0, 0), W_k.real[0,0,:], label="Gaussian filter in Fourier")
# plt.title("Gaussian filter in real v fourier")
# plt.legend()
# -

# Pksmooth = PKL.Pk(smoothed_field, BoxSize, 0, MAS, threads, verbose)
# # 3D P(k)
# ksmooth       = Pksmooth.k3D
# Pk0smooth     = Pksmooth.Pk[:,0] #monopole


# +
# grid_size = W_k.shape[0]
# kx, ky, kz = np.meshgrid(np.fft.fftfreq(grid_size), np.fft.fftfreq(grid_size), np.fft.rfftfreq(grid_size))
# k_magnitude = np.sqrt(kx**2 + ky**2 + kz**2)
# unique_magnitudes = np.unique(k_magnitude)
# W_1D = np.zeros(len(unique_magnitudes), dtype=np.complex128)

# for i, mag in enumerate(unique_magnitudes[:100]):
#     indices = np.where(k_magnitude == mag)
#     W_1D[i] = np.mean(W_k[indices])
#     if i%100==0:
#         print(i)

# +
# plt.plot(unique_magnitudes[:100], W_1D[:100])
# plt.plot(unique_magnitudes[:100], np.exp(-unique_magnitudes[:100]**2*R), label="Gaussian filter in Fourier")


# plt.legend()


# +
# # Right now I am smoothing out a lot of the detail that ideally I want to capture, up to k=60.
# # I will smooth it less next time
# # There is also a window function kicking in ~k=5

# power_mod = np.exp(-k3ds**2*R)**2


# plt.plot(k3ds, P(k3ds), label='P(k) input')
# plt.plot(k, Pk0, label='P(k) measured (no smooth)')
# plt.plot(ksmooth, Pk0smooth, label='P(k) measured (smooth)')
# plt.plot(k3ds, P(k3ds)*power_mod, label='P(k) input after modification')
# plt.yscale('log')
# plt.xscale('log')
# plt.ylim([10**-6,10])
# plt.ylabel("P(k) [Mpc]")
# plt.xlabel("k [1/Mpc]")
# plt.legend()
# plt.axvline(.1/.7, color='k')
# plt.axvline(12/.7/2, color='k')

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
# -

L/npix_perside*2

# +
nskew_per_side = 675/2
print(nskew_per_side)
ix = np.arange(int(npix_perside//nskew_per_side), df_3D.shape[0], int(npix_perside//nskew_per_side))
iy = np.arange(int(npix_perside//nskew_per_side), df_3D.shape[0], int(npix_perside//nskew_per_side))

points    = np.array([(x, y) for x in ix for y in iy])
positions = points * L / df_3D.shape[0]
# -

plt.imshow(df_3D[:,:,1])
plt.scatter(points[:,0][::10], points[:,1][::10], color='k', s=2)

# option 2: compute random skewers
# thetabin_centers = [0.290, 0.972, 2.204, 3.444, 5.941]
# thetabins_Mpc    = [[tbin-0.1, tbin+0.1] for tbin in thetabin_centers] # update this to be variable
# # find all skewer pairs separated by an amount within the bin
# power_bins = []
# # first, add the 0-separated skewers
# inbin = distances==0
# print("Number of 'pairs' separated by 0", np.sum(inbin))
# power_inbin = np.average(np.real(delta_flux_k[pair_indices[inbin]][0,:]*np.conjugate(delta_flux_k[pair_indices[inbin]][1,:])), axis=0)
# power_bins.append(power_inbin)
# for tbin in thetabins_Mpc:
#     print(tbin)
#     inbin = (distances<tbin[1]) & (distances>tbin[0])
#     print("Number of pairs in this bin", np.sum(inbin))
#     # compute the cross-power
#     power_inbin = np.average(np.real(delta_flux_k[pair_indices[inbin]][0,:]*np.conjugate(delta_flux_k[pair_indices[inbin]][1,:])), axis=0)
#     power_bins.append(power_inbin)


# skewers_ax1 = df_3D[points[:,0], points[:,1], :] # do with unsmoothed field
# skewers_ax2 = df_3D[:, points[:,0], points[:,1]].T # do with unsmoothed field
# skewers_ax3 = df_3D[points[:,0], :, points[:,1]] # do with unsmoothed field
skewers_ax1 = df_3D[points[:,0], points[:,1], :] # do with smooth field
skewers_ax2 = df_3D[:, points[:,0], points[:,1]].T # do with smooth field
skewers_ax3 = df_3D[points[:,0], :, points[:,1]] # do with smooth field


print(skewers_ax1.shape, skewers_ax2.shape, skewers_ax3.shape)
for skewer in skewers_ax1[:5]:
    plt.plot(skewer)

# +
Ns,Np = skewers_ax1.shape

print(Ns, Np)
size = int(np.sqrt(Ns))
position_grid  = positions.reshape((size,size,2))
skewers_ax1 = skewers_ax1.reshape((size,size,Np))
skewers_ax2 = skewers_ax2.reshape((size,size,Np))
skewers_ax3 = skewers_ax3.reshape((size,size,Np))
delta_flux_k1 = np.fft.rfft(skewers_ax1) # Fourier transform all the skewers
delta_flux_k2 = np.fft.rfft(skewers_ax2) # Fourier transform all the skewers
delta_flux_k3 = np.fft.rfft(skewers_ax3) # Fourier transform all the skewers
# -

print(skewers_ax1.shape, skewers_ax2.shape, skewers_ax3.shape)
for skewer in skewers_ax1[0, 0:5, :]:
    plt.plot(skewer)

dz = np.linalg.norm(position_grid[0,1]-position_grid[0,0])
print(dz) # actual Mpc separation between positions
spacing = [0,0.5, 1, 2, 4, 7]
# spacing_dL = np.array([0,1,2,4,6]) # the integer number of dz that corresponds to that spacing
# spacing = spacing_dL*dz
spacing_dL = np.round(spacing / dz,0).astype(int)
print("desired spacing") # the actual spacings between skewers
print("integral number of transverse skewer spacings in each dL", spacing_dL)
actual_spacing = dz*spacing_dL
print("actual spacing", actual_spacing)

# +
Px = []
Px_errs = []

for s,dL in enumerate(spacing_dL):
    Px_dL = []
    Px_dL_errs = []
    print("spacing = ", spacing_dL[s])
    for i in range(0, size):
        
        for j in range(0, size-dL, dL+1):
        
            # Px_dL.append(np.real(delta_flux_k1[i,j]*np.conjugate(delta_flux_k1[i,j+dL]))) # loop through 0 axis
            # if dL!=0:
            #     Px_dL.extend(np.real(delta_flux_k1[i,j]*np.conjugate(delta_flux_k1[i+dL,j]))) # loop through 1 axis
            # Px_dL.append(np.real(delta_flux_k2[i,j]*np.conjugate(delta_flux_k2[i,j+dL]))) # loop through 0 axis
            # if dL!=0:
            #     Px_dL.extend(np.real(delta_flux_k2[i,j]*np.conjugate(delta_flux_k2[i+dL,j]))) # loop through 1 axis
            Px_dL.append(np.real(delta_flux_k3[i,j]*np.conjugate(delta_flux_k3[i,j+dL]))) # loop through 0 axis
            # if dL!=0:
            #     Px_dL.extend(np.real(delta_flux_k3[i,j]*np.conjugate(delta_flux_k3[i+dL,j]))) # loop through 1 axis
            # need to account here for periodic boundaries
            if i==20 and j==20:
                print('position diff', np.linalg.norm(position_grid[i,j] - position_grid[i,j+dL]))
                print(np.linalg.norm(position_grid[i,j] - position_grid[i+dL,j]))
        
        
    print(len(Px_dL))
    print(Px_dL[0].shape)

    # after looping through, average all the results
    Px_dL = np.asarray(Px_dL)
    avg_over_phases = np.average(Px_dL, axis=0)*(L/(Np**2))
    print(avg_over_phases.shape)
    Px.append(avg_over_phases)
    print(len(Px_dL))
    errs = np.std(Px_dL, axis=0)*(L/(Np**2))/np.sqrt(len(Px_dL))
    Px_errs.append(errs)
# -

kpar = np.fft.rfftfreq(Np, pix_size)*2*np.pi # frequency in Mpc^-1

# ## Make the predictions using brute-force integral

kmin = 2*np.pi/L

# +
# from scipy import special
# Nsteps = 2**10

# # for every rperp of separation between sightlines
# # make an empty list for each rperp
# Px_rperp = []

# for rperp in spacing:
#     print("r perp:", rperp)
#     # make an empty list for each k parallel
#     Px_rperp_kpar = []
#     for c, kp in enumerate(kpar[1:]):
#         # initialize integral
#         Px_int = 0
#         kmax_int = 10000 # maximum k where I'll cut off the integrals -- don't worry if it's too big because I continue later if it gets too large
#         kvar = np.logspace(np.log10(kp+.001), np.log10(kmax_int), int(Nsteps-c))
#         for m,k in enumerate(kvar[:-1]):
#             # get kperp given k and kpar
#             if (k**2-kp**2)<0:
#                 print("problem")
#                 print(k, kp)
#                 break
#             kperp = np.sqrt(k**2-kp**2)
#             if rperp!=0 and (kperp>2*np.pi/rperp):
#                 continue # Px cannot sample perp modes smaller in wavelength than the spacing
#             elif (kperp<kmin) or (k<kmin):
#             #     print("too small kperp or k:", kperp, k)
#                 continue
#             # get mu
#             mu = kpar/k
#             # get the bessel function j0 = sqrt(pi/(2x))* J_(1/2) evaluated at k_perp * r_perp
#             j0 = special.jv(0, kperp * rperp)
#             # get P3D from model at z, k, mu
#             P3D = P_iso(0, np.asarray([k]), smooth=True, smoothing=smoothing)
#             # add to integral
#             dlogk = kvar[m+1]-kvar[m] # (kmax_int-kpar)/int(Nsteps-kpar)
#             Px_int += j0*P3D*dlogk*k
#         Px_int /= (2*np.pi)
#         Px_rperp_kpar.append(Px_int)
#     Px_rperp.append(Px_rperp_kpar)
# Px_rperp_funkyint = Px_rperp


# +
# from scipy import special
# Nsteps = 2**10
# int_range = np.logspace(np.log10(kmin), 1, Nsteps)
# # for every rperp of separation between sightlines
# # make an empty list for each rperp
# Px_rperp = []
# for rperp in spacing:
#     # make an empty list for each k parallel
#     Px_rperp_kpar = []
#     print("r perp:", rperp)
#     for kp in kpar:
#         # initialize integral
#         Px_int_kpar = 0
#         for kperp in int_range:
#             dlogkperp = int_range[m+1]-int_range[m] # (kmax_int-kpar)/int(Nsteps-kpar)
#             J0 = special.jv(0, kperp * rperp)
#             # get P3D from model at kperp, kpar
#             k = np.sqrt(kperp**2 + kp**2)
#             P3D = P_iso(0, np.asarray([k]), smooth=True, smoothing=smoothing)
#             # add to integral
#             Px_int_kpar += dlogkperp*J0*kperp*P3D
#         Px_int_kpar /= (2*np.pi)
#         Px_rperp_kpar.append(Px_int_kpar)
#     Px_rperp.append(Px_rperp_kpar)


# -

# from scipy import special
# Nsteps=2**12
# int_range = np.logspace(np.log10(kmin), 1, Nsteps)
# # for every rperp of separation between sightlines
# # make an empty list for each rperp
# Px_rperp = []
# for rperp in spacing:
#     Px_per_kpar = []
#     for kp in kpar:
#         kperp  = np.logspace(np.log10(kmin), 1, Nsteps)
#         kpfull = np.full(len(kperp), kp)
#         k = np.sqrt(kperp**2 + kpfull**2)
#         J0 = special.jv(0, kperp*rperp)
#         y = J0*kperp*P_iso(0, k, smooth=True, smoothing=smoothing)
#         Px_per_kpar.append(np.trapz(y=y, x=kperp))
#     Px_rperp.append(np.asarray(Px_per_kpar)/(2*np.pi))
        

# +
# for r in range(len(spacing)):
#     plt.semilogx(kpar,np.asarray(Px_rperp[r]),  label=r'$r_{{\perp}}={:.2f}$'.format(spacing[r]), color=cmap(r/len(spacing)))
# plt.legend()


# plt.xlim([.4, 14])
# # plt.ylim([10**-7,10**-1])
# plt.ylabel(r"$P_{\times} [h^{-1}$ Mpc]")
# plt.xlabel(r"$k_{\parallel} [h$ Mpc$^{-1}$]")
# # p1d = arinyo.P1D_Mpc(zs[0], kpars, parameters=params)
# # plt.loglog(kpar, p1d, label='arinyo model 1D')
# -

# ## Make the predictions using Hankel transform

# +
# # now get the corresponding Pcross predictions

# import numpy as np


def get_Px_nonmod(
    kpars,
    P3D,
    fast=False,
    min_rperp=0.01,
    max_rperp=30,
    min_kperp=10.0**-20,
    max_kperp=10.0**3,
    Nsteps_kperp=2**13,
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
    if min_rperp is not None and min_rperp > 0.08:
        trans_to_p1d = False  # not necessary to transition to the P1D result if minimum requested rperp is larger than 0.08

    if min_rperp is not None and min_rperp > 0.2 and fast:
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
        func = P3D(0, k, smooth=True, smoothing=smoothing) * kperps
        rperp, LHS = hankl.FFTLog(
            kperps, func, q=0, mu=0
        )  # returns an array of log-spaced rperps, and the Hankel Transform
        Px = LHS / rperp / (2*np.pi)  # Divide out by remaining factor to get Px
        if min_rperp is not None and min_rperp > min(rperp):
            rperp_minidx = np.argmin(abs(rperp - min_rperp))
        
        if max_rperp is not None and max_rperp < max(rperp):
            rperp_maxidx = np.argmin(abs(rperp - max_rperp))
        else:
            rperp_minidx, rperp_maxidx = None, None
        Px_per_kpar.append(Px[rperp_minidx:rperp_maxidx])
        rperp = rperp[rperp_minidx:rperp_maxidx]
    Px_per_kpar = np.asarray(Px_per_kpar)

    return rperp, Px_per_kpar



# -

from forestflow.pcross import get_Px

# +
# rperp,Px_per_kpar = get_Px(kpar, P, min_rperp=None, max_rperp=None)
# rperp, Px_per_kpar = get_Px(
#     kpar,
#     P,
#     0,
#     P3D_mode='cart',
#     )
# Px_per_rperp = Px_per_kpar.T

rperp, Px_per_kpar = get_Px(
    kpar,
    P_pol,
    0,
    P3D_mode='pol',
    **{"smooth":True, "smoothing":smoothing})
Px_per_rperp = Px_per_kpar.T
# find the closest predicted rperps to the measured skewer spacings
idxs = []
for i in range(len(actual_spacing)):
    idxs.append(np.argmin(abs(rperp-actual_spacing[i])))
    print(rperp[idxs[i]],actual_spacing[i])
    
# -

rperp.shape, Px_per_kpar.shape

# +
# tosave = np.column_stack((rperp,Px_per_kpar.T))
# np.savetxt(f"/nfs/pic.es/user/m/mlokken/Lya_Px/Px_res{pix_size}_L{L}.dat",tosave)
# -

#min frequency, max frequency
print(f"We can't expect to measure anything smaller than k={2*np.pi/L} or larger than k={np.pi/(L/Np)}")

# First, make sure P1D works
pk_avgs = []
for delta_k in [delta_flux_k1, delta_flux_k2, delta_flux_k3]:
    print(delta_k.shape)
    pk_all=abs(delta_k.reshape((Ns,len(kpar))))**2*(L/(Np**2)) # division by Np for avg power in all modes, convert power per unit k to power per unit spacing
    pk_avg_ax = np.average(pk_all, axis=0) # average over all skewers in this axis
    print(pk_avg_ax.shape)
    pk_avgs.append(pk_avg_ax)
pk_avg = np.average(np.asarray(pk_avgs), axis=0) # now average over all axes
print(pk_avg.shape)

# +
plt.plot(kpar,pk_avg, 'o', markersize=3, label='skewers 1D', linestyle='solid')
plt.errorbar(kpar,Px[0],Px_errs[0], label=f'spacing = {round(spacing[0],1)}, skewers smallest Px', color=cmap(i/len(spacing)), linestyle='dashed')

# plt.yscale("log")
plt.xscale("log")
plt.ylim([0.0,0.2])
plt.legend()
plt.xlabel(r"kpar [Mpc$^{-1}$]")
plt.ylabel("P1D [Mpc]")
# -

rperp_nonmod, Px_per_kpar_nonmod = get_Px_nonmod(kpar, P_iso)

plt.semilogx(kpar,pk_avg, 'o', markersize=3, label='skewers', linestyle='solid')
plt.semilogx(kpar,Px_per_kpar.T[0], label=f'spacing = {round(spacing[0],1)}, module', color=cmap(i/len(spacing)), linestyle='dotted')
# plt.semilogx(kpar,Px_rperp[0], label=f'spacing = {round(spacing[0],1)}, bf integral', color=cmap(i/len(spacing)), linestyle='dashed')
# plt.semilogx(kpar,Px_per_kpar_nonmod.T[0], label='non-module function', color='black')
plt.legend()
plt.ylim([0.0,0.05])

plt.plot(rperp, Px_per_kpar[0])
plt.xscale("log")

plt.semilogx(rperp,Px_per_kpar[0], 'o', markersize=3, linestyle='solid', label='Px integral')
plt.semilogx(np.logspace(np.log10(min(rperp)), np.log10(max(rperp)), len(rperp)), np.full(len(rperp), pk_avg[0]), label='P1D measured kpar~0') # plot P1d for the lowest k
plt.xlabel("rperp")
plt.ylabel("Px")
# plt.errorbar(kpar,Px[0],Px_errs[0], label=f'spacing = {round(spacing[0],1)}, skewers smallest Px', color=cmap(i/len(spacing)), linestyle='dashed')
# # plt.yscale("log")
# plt.xscale("log")
# # plt.ylim([0.14,0.16])
plt.legend()

# +
# colors=['red', 'cornflowerblue', 'purple', 'brown', 'green', 'black', 'orange', 'yellow', ]
# -

# # Do the predictions from Hankel and brute force match?

# +
# for i in range(len(spacing)):
#     plt.semilogx(kpar, Px_per_kpar.T[idxs[i]], label=f'Hankel rperp = {round(actual_spacing[i],1)}, integral', color=cmap(i/len(spacing)))
#     if i==0:
#         labelb = 'brute force polar'
#         labelc = 'brute force cart'
#     else:
#         labelb = ''
#         labelc = ''
#     plt.semilogx(kpar[1:],np.asarray(Px_rperp_funkyint[i]), label=labelb, color=cmap(i/len(spacing)), linestyle='dotted')
#     plt.semilogx(kpar, np.asarray(Px_rperp[i]), label=labelc, color=cmap(i/len(spacing)), linestyle='dashed')
# plt.legend()
# plt.xlim([.4, 14])
# # plt.ylim([10**-7,0.045])
# plt.ylabel(r"$P_{\times}$ [Mpc]")
# plt.xlabel(r"$k_{\parallel} [$ Mpc$^{-1}$]")
# # p1d = arinyo.P1D_Mpc(zs[0], kpars, parameters=params)
# # plt.loglog(kpar, p1d, label='arinyo model 1D')

# +

# for i in range(len(spacing[:5])):
#     plt.errorbar(kpar, Px[i], yerr=Px_errs[i], label=f'rperp = {round(actual_spacing[i],1)}Mpc, skewers', color=cmap(i/len(spacing)), linestyle='dashed')
#     if i==1:
#         intlabel1 = 'Integral Predictions, brute force'
#         intlabel2 = 'brute force int2'
#         intlabel3 = 'Hankel'
#     else:
#         intlabel1=''
#         intlabel2=''
#         intlabel3=''
        
#     plt.plot(kpar, np.asarray(Px_rperp[i]), label=intlabel1, color=cmap(i/len(spacing)), linestyle='dashed')
#     plt.plot(kpar[1:], np.asarray(Px_rperp_funkyint[i]), label=intlabel2, color=cmap(i/len(spacing)), linestyle='dotted')
#     plt.plot(kpar, np.asarray(Px_per_kpar.T[idxs[i]]), label=intlabel3, color=cmap(i/len(spacing)), linestyle='solid')
    
# # plt.yscale('log')
# plt.xscale('log')
# plt.legend()
# plt.xlim([0.1,2])
# # plt.ylim([-0.003,0.047])
# plt.title(f"$L={L}, \Delta_{{p}}={np.round(pix_size,2)}$ Mpc")
# plt.xlabel("kpar [Mpc-1]")
# plt.ylabel("Px [Mpc]")



# +
fig,ax = plt.subplots(nrows=2,ncols=1, figsize=[8,5], gridspec_kw={'height_ratios':[3,1]}, sharex=True)

for i in range(len(spacing)):
    # plt.plot(kpar, Px[i], label=f'spacing = {round(spacing[i],1)}, skewers', color=cmap(i/len(spacing)), linestyle='dashed')
    # plt.errorbar(kpar, Px[i], yerr=Px_errs[i], label=f'rperp = {round(spacing[i],1)}, skewers', color=cmap(i/len(spacing)), linestyle='dashed')
    ax[0].errorbar(kpar, Px[i], yerr=Px_errs[i], label=f'rperp = {round(actual_spacing[i],1)}Mpc, skewers', color=cmap(i/len(spacing)), linestyle='dashed')

    # plt.plot(kpar, Px_per_kpar.T[idxs[i]], label=f'rperp = {round(spacing[i],1)}, integral', color=cmap(i/len(spacing)))
    if i==1:
        intlabel = 'Integral Predictions, Hankel'
    else:
        intlabel=''
    ax[0].plot(kpar, Px_per_kpar.T[idxs[i]], label=intlabel, color=cmap(i/len(spacing)))
    pctdiff = ((Px[i]-Px_per_kpar.T[idxs[i]])/Px[i])*100
    absdiff = (Px[i]-Px_per_kpar.T[idxs[i]])
    # ax[1].plot(kpar, pctdiff, c=cmap(i/len(spacing)))
    ax[1].errorbar(kpar, absdiff, yerr=Px_errs[i], c=cmap(i/len(spacing)), linestyle='dashed')
    
# plt.yscale('log')
ax[0].set_xscale('log')
ax[0].legend()
ax[0].set_xlim([0.1,10])
ax[0].set_ylim([-0.003,0.08])
plt.suptitle(f"$L={L}, \Delta_{{p}}={np.round(pix_size,2)}$ Mpc")
ax[1].set_xlabel("kpar [Mpc-1]")
ax[0].set_ylabel("Px [Mpc]")
ax[1].set_ylim([-.01,.01])

ax[1].set_ylabel("meas-pred")
# -

2*np.pi/7




