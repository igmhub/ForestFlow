import numpy as np

def generate_grf(boxlen_mpc, power_spectrum_func, npix_per_side = 1000):
    from numpy.fft import rfftn, irfftn

    pixel_size_mpc = boxlen_mpc / npix_per_side
    shape = (npix_per_side, npix_per_side, npix_per_side)
    # First, create a 3D grid of spatial frequencies
    kx = 2 * np.pi * np.fft.fftfreq(shape[0], pixel_size_mpc)
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], pixel_size_mpc)
    kz = 2 * np.pi * np.fft.rfftfreq(shape[2], pixel_size_mpc)
    kgrid = np.meshgrid(kx, ky, kz, indexing='ij')
    # Calculate the radial distance from the origin in frequency space
    k = np.sqrt(kgrid[0]**2 + kgrid[1]**2 + kgrid[2]**2)
    print(k.shape)
    
    # Step 2: Generate a complex field with the desired power spectrum
    freqshape = (shape[1],shape[2], shape[0]//2+1)
    # freqshape = shape
    field_in_freq = np.random.normal(0, 1, size=freqshape) + 1j*np.random.normal(0, 1, size=freqshape)
    field_in_freq *= np.sqrt(0.5*power_spectrum_func(k))
    field_in_freq[0,0,0] = 0 # set the mean to 0
    # field_in_freq[-1, -1, -1] = modes[-1,-1,-1].real # check this

    # Step 3: Inverse Fourier Transform to get the real-space field
    field_in_real_space = irfftn(field_in_freq) # need to normalize this after

    return field_in_real_space
