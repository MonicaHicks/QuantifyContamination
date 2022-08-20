from astropy import units as u
import healpy as hp
import numpy as np
import pandas as pd

d9 = hp.read_map('dust_maps/d9_2048_353_I.fits')
d10 = hp.read_map('dust_maps/d10_2048_353_I.fits')
d11 = hp.read_map('dust_maps/d11_2048_353_I.fits')
d12 = hp.read_map('dust_maps/d12_2048_353_I.fits')

dust_maps = [d9,d10,d11,d12]

galaxy_data = pd.read_csv('GLADE+_2048.csv')
pixels = galaxy_data.Pix
z = galaxy_data.Z
z_bin_i = galaxy_data.Z_bin

tomo = pd.read_csv('Tomographer_GLADE+.csv')
tomo_z = tomo['z']
tomo_plots = tomo['dNdz_b']
z_bins = tomo_z

smooth_dust_maps = []

for i in range(len(dust_maps)):
	smooth_dust_maps.append(hp.sphtfunc.smoothing(dust_maps[i],fwhm=0.0174533,iter=1))

z_maps = []
nside = 2048
npixels = 12*nside**2
n_zbins = len(z_bins)
bins = np.arange(npixels + 1)

for i in range(n_zbins):
    pix_for_zbin_i_gal = pixels[z_bin_i == i]
    N_galaxies_in_zbin_i_in_each_pixel, bin_edges = np.histogram(pix_for_zbin_i_gal, bins=bins)
    z_maps.append(N_galaxies_in_zbin_i_in_each_pixel)

smooth_z_maps = []
# make parallel array of z_maps smoothed to 1 square degree
for i in range(len(z_maps)):
    smooth_z_maps.append(hp.sphtfunc.smoothing(z_maps[i], fwhm=0.5556, iter=1))

# pixel index of every pixel in your map
pixindx = np.arange(npixels)

# get array of (l, b) Galactic longitude and latitude values for every pixel
l_pix, b_pix = hp.pix2ang(nside, pixindx, lonlat=True)

# set everything below an absolute value of the Galactic latitude 30 degrees to 0 in a mask
mask_pixels = pixindx[np.abs(b_pix) < 30.]

corr_red = []

for i in range(len(dust_maps)):
	corr_red.append(np.subtract(dust_maps[i][mask_pixels],smooth_dust_maps[i][mask_pixels]))

corr_bins = []
ones_arr = np.ones(len(mask_pixels))

for i in range(n_zbins):
	exp_galaxy_count = np.subtract(smooth_z_maps[i][mask_pixels],ones_arr)
	act_galaxy_count = z_maps[i][mask_pixels]
	corr_bins.append(np.divide(act_galaxy_count,exp_galaxy_count))

corr_data = [[],[],[],[]]

for i in range(len(dust_maps)):
	for j in range(len(corr_bins)):
		corr_data[i].append(int(np.correlate(corr_bins[j],corr_red[i])))

df = pd.DataFrame()
df['z_bins'] = tomo_z
df['d9'] = corr_data[0]
df['d10'] = corr_data[1]
df['d11'] = corr_data[2]
df['d12'] = corr_data[3]
df.to_csv('d9-d12_353_corr_data_output.csv')
