from astropy import units as u
import healpy as hp
import numpy as np
import pandas as pd
import pysm3
from pysm3.models.dust import blackbody_ratio

# set map resolution and frequency
nside = 2048
freq = 353 * u.GHz


# generate dust intensity maps using PySM
dust_models = ['d0','d1','d2','d3','d4','d5','d7','d8','d9','d10','d11','d12']
dust_i_maps = []

for i in range(len(dust_models)):
	print(dust_models[i])
	if dust_models[i] == 'd11':
		break
	sky_model = pysm3.Sky(preset_strings=[dust_models[i]],nside=nside)
	dust_i_maps.append(sky_model.get_emission(freq)[0])

d11_config = pysm3.sky.PRESET_MODELS['d11'].copy()
del d11_config['class']
sky_model = pysm3.models.ModifiedBlackBodyRealization(nside=nside, **d11_config)
dust_i_maps.append(sky_model.get_emission(freq)[0])


d12_config = pysm3.sky.PRESET_MODELS['d12'].copy()
del d12_config['class']
sky_model = pysm3.models.ModifiedBlackBodyLayers(nside=nside, **d12_config)
dust_i_maps.append(sky_model.get_emission(freq)[0])


# pixel index of every pixel in your map
pixindx = np.arange(12*nside**2)

# get array of (l, b) Galactic longitude and latitude values for every pixel
l_pix, b_pix = hp.pix2ang(nside, pixindx, lonlat=True)

# set everything below an absolute value of the Galactic latitude 30 degrees to 0 in a mask
mask_pixels = pixindx[np.abs(b_pix) < 30.]
good_pixels = pixindx[np.abs(b_pix) > 30.]

# read ~22 million galaxy and quasar data
galaxy_data = pd.read_csv('/scratch/users/mdhicks/pysm/GLADE+_2048.csv')
galaxy_data = galaxy_data.loc[(galaxy_data.Z > 0) & (galaxy_data.Z < 3.1488)]
galaxy_data = galaxy_data.loc[(galaxy_data.Pix < mask_pixels[0]) | (galaxy_data.Pix > mask_pixels[-1])]
pixels = galaxy_data.Pix
z = galaxy_data.Z
z_bin_i = galaxy_data.Z_bin


# get z-bins from tomographer data
tomo = pd.read_csv('/scratch/users/mdhicks/pysm/Tomographer_GLADE+.csv')
tomo_z = tomo['z']
tomo_plots = tomo['dNdz_b']
z_bins = tomo_z


# generate smoothed dust maps to simulate 1-degree running mean of neighbors
smooth_dust_maps = []

for i in range(len(dust_i_maps)):
	smooth_dust_maps.append(hp.sphtfunc.smoothing(dust_i_maps[i],fwhm=0.0174533,iter=1))


# generate galaxy density maps by z-bin
z_maps = []
npixels = 12*nside**2
n_zbins = len(z_bins)
bins = np.arange(npixels + 1)

for i in range(n_zbins):
    pix_for_zbin_i_gal = pixels[z_bin_i == i]
    N_galaxies_in_zbin_i_in_each_pixel, bin_edges = np.histogram(pix_for_zbin_i_gal, bins=bins)
    z_maps.append(N_galaxies_in_zbin_i_in_each_pixel)


# make parallel array of z-bin galaxy density maps smoothed to 1 square degree
smooth_z_maps = []

for i in range(len(z_maps)):
    smooth_z_maps.append(hp.sphtfunc.smoothing(z_maps[i], fwhm=0.562, iter=1))


# calculate E_{B-V} - <E_{B-V}> for each dust models
corr_red = []

for i in range(len(dust_i_maps)):
	val = dust_i_maps[i].value - smooth_dust_maps[i]
	corr_red.append(val[good_pixels])


# calculate N / (<N> - 1) for each z-bin galaxy density map
corr_bins = []
ones_arr = np.ones(len(good_pixels))

for i in range(n_zbins):
	exp_galaxy_count = np.subtract(smooth_z_maps[i][good_pixels],ones_arr)
	act_galaxy_count = z_maps[i][good_pixels]
	corr_bins.append(np.divide(act_galaxy_count,exp_galaxy_count))


# cross correlate each galaxy density map with each dust intensity map
corr_data = []

for i in range(len(dust_models)):
	corr_data.append([])

for i in range(len(dust_i_maps)):
	for j in range(len(corr_bins)):
		corr_data[i].append(int(np.correlate(corr_bins[j],corr_red[i])))


# save results to csv file with one column for the result for each dust model
df = pd.DataFrame()
df['z_bins'] = tomo_z

for i in range(len(dust_models)):
	df[dust_models[i]] = corr_data[i]

df.to_csv('d1-d12_contamination_353.csv')
