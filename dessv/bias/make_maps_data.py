"""
This script runs through and makes all maps needed for the 
bias calculation. The final product is a 4D matrix for kappa_g 
and gamma_g, 3D for kappa and gamma.

Note for kappa_g and gamma_g maps, we also make 100 randoms 
per map.
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np 
import pyfits as pf 
from numpy import random 
sys.path.append('../utils')
from kappag import *
from massmapping_utils import *

import config_data
for name in [name for name in dir(config_data) if not name.startswith("__")]:
    globals()[name] = getattr(config_data, name)

random.seed(ran_seed)

# load data and mask #######

print('load data...')
mask_map = np.load(temp_dir_data+'mask_map.npz')['mask']
mask_bias = np.load(temp_dir_data+'mask_bias.npz')['mask']
cat_bench = pf.open(temp_dir_data+'benchmark_masked.fits')[1].data
mask_fraction = np.load(temp_dir_data+'mask_fraction.npz')['fraction']

ra = cat_bench['RA']
dec = cat_bench['DEC']
# pofz = cat_gold_pofz['ZPDF_BIN']
zmean = cat_bench['ZMEAN_'+str(photoz)]

ran = pf.open(ran_name)[1].data
ran_len = len(ran)

# make kappa_g/gamma_g maps ########

print('make kappa_g/gamma_g maps...')
ra_p = ra_ref + (ra-ra_ref)*np.cos(dec/180.*np.pi)
kg_4d = np.zeros((Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))
kg_ran_4d = np.zeros((N_ran, Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))
k2e1_4d = np.zeros((Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))
k2e2_4d = np.zeros((Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))
k2e1_ran_4d = np.zeros((N_ran, Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))
k2e2_ran_4d = np.zeros((N_ran, Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))

# first loop through lens bins
for i in range(Nbin_z_l):
	print('lens bin'+str(i))

	# # if use full pofz for each galaxy
	# if nofz_type==1:
	# 	nofz_l = np.load('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/photoz_corr/nofz_lens_2.npz')['nofz']
	# 	nofz_s = np.load('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/photoz_corr/nofz_source_2.npz')['nofz'] 
	# 	mask = ((z_pofz_bin+0.05)>Z1_l[i])*((z_pofz_bin+0.05)<=Z2_l[i])
	# 	# first calculate the weighting of each galaxy according to its pdf    
	# 	n_weight = np.zeros(len(dec))
	# 	for j in range(len(z_pofz_bin[mask])):
	# 		n_weight += pofz[:, int(z_pofz_bin[mask][j]/0.1)]

	#     # then loop through source bins
	# 	for j in range(Nbin_z_s):
	# 		print('source bin'+str(j))
	# 		if j>=i:
	# 			N2d_bin, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=n_weight)
	# 			#kg_4d[j][i] = kappag_map_bin_nofz(N2d_bin, mask_map, nofz_l[i], nofz_s[j], cosmo, smooth_kernal, smooth_scale)
	# 			kg_4d[j][i] = kappag_map_bin(N2d_bin, mask_map, Z1_l[i], Z2_l[i], z_s[j], cosmo, smooth_kernal, smooth_scale)
	# 			k2e1, k2e2 = k2g_fft(kg_4d[j][i], kg_4d[j][i]*0.0, pix, pad=True)
	# 			k2e1_4d[j][i] = k2e1
	# 			k2e2_4d[j][i] = k2e2			

	# 			# make the random maps
	# 			Ngal = len(dec)
	# 			for k in range(N_ran):
	# 				ran_ids = random.choice(range(ran_len), size=Ngal, replace=False)
	# 				ra_ran = ran['RA'][ran_ids]
	# 				dec_ran = ran['DEC'][ran_ids]
	# 				ra_ran_p = ra_ref + (ra_ran-ra_ref)*np.cos(dec_ran/180.*np.pi)
	# 				N2d_bin, edges = np.histogramdd(np.array([dec_ran, ra_ran_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=n_weight)
	# 				#kg_ran_4d[k][j][i] = kappag_map_bin_nofz(N2d_bin, mask_map, nofz_l[i], nofz_s[j], cosmo, smooth_kernal, smooth_scale)
	# 				kg_ran_4d[k][j][i] = kappag_map_bin(N2d_bin, mask_map, Z1_l[i], Z2_l[i], z_s[j], cosmo, smooth_kernal, smooth_scale)
	# 				k2e1, k2e2 = k2g_fft(kg_ran_4d[k][j][i], kg_ran_4d[k][j][i]*0.0, pix, pad=True)
	# 				k2e1_ran_4d[k][j][i] = k2e1
	# 				k2e2_ran_4d[k][j][i] = k2e2	

	# np.savez(temp_dir_data+'kg_4d_data_pofz.npz', kg=kg_4d, kg_ran=kg_ran_4d, k2e1=k2e1_4d, k2e2=k2e2_4d, k2e1_ran=k2e1_ran_4d, k2e2_ran=k2e2_ran_4d)

	# if use full zmean for each galaxy
	if zmean_type==1:
		zmask = (zmean>Z1_l[i])*(zmean<=Z2_l[i])
		# first calculate the weighting of each galaxy according to its pdf    
		# nofz_l = np.load('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/photoz_corr/nofz_lens.npz')['nofz']
		# nofz_s = np.load('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/photoz_corr/nofz_source.npz')['nofz'] 
		n_weight = dec[zmask]*0.0+1.0

	    # then loop through source bins
		for j in range(Nbin_z_s):
			print('source bin'+str(j))
			if j>=i:
				N2d_bin, edges = np.histogramdd(np.array([dec[zmask], ra_p[zmask]]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=n_weight)
				#kg_4d[j][i] = kappag_map_bin_nofz(N2d_bin, mask_map, nofz_l[i], nofz_s[j], cosmo, smooth_kernal, smooth_scale)
				kg_4d[j][i] = kappag_map_bin(N2d_bin, mask_map, mask_fraction, Z1_l[i], Z2_l[i], z_s[j], cosmo, smooth_kernal, smooth_scale)

				k2e1, k2e2 = k2g_fft(kg_4d[j][i], kg_4d[j][i]*0.0, pix, pad=True)
				k2e1_4d[j][i] = k2e1
				k2e2_4d[j][i] = k2e2			

				# make the random maps
				Ngal = len(dec[zmask])
				for k in range(N_ran):
					ran_ids = random.choice(range(ran_len), size=Ngal, replace=False)
					ra_ran = ran['RA'][ran_ids]
					dec_ran = ran['DEC'][ran_ids]
					ra_ran_p = ra_ref + (ra_ran-ra_ref)*np.cos(dec_ran/180.*np.pi)
					N2d_bin, edges = np.histogramdd(np.array([dec_ran, ra_ran_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=n_weight)
					#kg_ran_4d[k][j][i] = kappag_map_bin_nofz(N2d_bin, mask_map, nofz_l[i], nofz_s[j], cosmo, smooth_kernal, smooth_scale)
					kg_ran_4d[k][j][i] = kappag_map_bin(N2d_bin, mask_map, mask_fraction, Z1_l[i], Z2_l[i], z_s[j], cosmo, smooth_kernal, smooth_scale)

					k2e1, k2e2 = k2g_fft(kg_ran_4d[k][j][i], kg_ran_4d[k][j][i]*0.0, pix, pad=True)
					k2e1_ran_4d[k][j][i] = k2e1
					k2e2_ran_4d[k][j][i] = k2e2	

	np.savez(temp_dir_data+'kg_4d_data_zmean.npz', kg=kg_4d, kg_ran=kg_ran_4d, k2e1=k2e1_4d, k2e2=k2e2_4d, k2e1_ran=k2e1_ran_4d, k2e2_ran=k2e2_ran_4d)


# make kappa/gamma maps ########

print('make kappa/gamma maps...')
e1_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))
e2_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))
e2kE_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))
e2kB_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))

e1_ran_3d = np.zeros((N_ran, Nbin_z_s, Nbin_dec, Nbin_ra))
e2_ran_3d = np.zeros((N_ran, Nbin_z_s, Nbin_dec, Nbin_ra))
e2kE_ran_3d = np.zeros((N_ran, Nbin_z_s, Nbin_dec, Nbin_ra))
e2kB_ran_3d = np.zeros((N_ran, Nbin_z_s, Nbin_dec, Nbin_ra))

for i in range(Nbin_z_s):
	print('source bin'+str(i))

	if zmean_type==1:
		D = pf.open(temp_dir_data+'background_'+str(shear)+'_tomo_zmean_'+str(photoz)+'_'+str(i)+'.fits')[1].data
		nz_weight = D['RA']*0.0+1.0
	# if nofz_type==1:
	# 	D = pf.open(temp_dir_data+'background_'+str(shear)+'_tomo_pofz_'+str(photoz)+'_'+str(i)+'.fits')[1].data
	# 	nz_weight = D['NZ_weight']

	ra = D['RA']
	dec = D['DEC']
	e1 = D['S1']
	e2 = D['S2']
	w1 = D['W1']
	w2 = D['W2']

	ra_p = ra_ref + (ra-ra_ref)*np.cos(dec/180.*np.pi)

	N2d_e1, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=e1*w1*nz_weight)
	N2d_e2, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=e2*w2*nz_weight)
	N2d, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=nz_weight)
	N2d_w1, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=w1*nz_weight)
	N2d_w2, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=w2*nz_weight)

	N2dw1_patch = N2d_w1*1.0
	N2dw1_patch[N2dw1_patch==0]=1.0
	N2dw2_patch = N2d_w2*1.0
	N2dw2_patch[N2dw2_patch==0]=1.0
	N2d_e1 /= N2dw1_patch
	N2d_e2 /= N2dw2_patch

	N2d_s = N2d*1.0
	N2d_e1_s = N2d_e1*1.0 
	N2d_e2_s = N2d_e2*1.0 

	if (smooth>0.0):
	    N2d_s = smooth_map(N2d_s, mask_map, smooth_kernal, smooth_scale, 0.0) 
	    N2d_e1_s = smooth_map(N2d_e1_s, mask_map, smooth_kernal, smooth_scale, 0.0) 
	    N2d_e2_s = smooth_map(N2d_e2_s, mask_map, smooth_kernal, smooth_scale, 0.0) 

	# conversion
	e2kappaE, e2kappaB = g2k_fft(N2d_e1_s, N2d_e2_s, pix, pad=True)
	
	e1_3d[i] = N2d_e1_s
	e2_3d[i] = N2d_e2_s
	e2kE_3d[i] = e2kappaE
	e2kB_3d[i] = e2kappaB

	orig_ids = np.arange(len(dec))
	for k in range(N_ran):
		print(k)
		random.seed(ran_seed+k)
		temp_id = orig_ids*1
		random.shuffle(temp_id)
		N2d_ran_e1, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=e1[temp_id])
		N2d_ran_e2, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=e2[temp_id])

		N2d_ran_e1 /= N2dw1_patch
		N2d_ran_e2 /= N2dw2_patch
		N2d_ran_e1_s = N2d_ran_e1*1.0 
		N2d_ran_e2_s = N2d_ran_e2*1.0 

		if (smooth>0.0):
			N2d_ran_e1_s = smooth_map(N2d_ran_e1_s, mask_map, smooth_kernal, smooth_scale, 0.0) 
			N2d_ran_e2_s = smooth_map(N2d_ran_e2_s, mask_map, smooth_kernal, smooth_scale, 0.0) 

		# conversion
		e2kappaE_ran, e2kappaB_ran = g2k_fft(N2d_ran_e1_s, N2d_ran_e2_s, pix, pad=True)
		
		e1_ran_3d[k][i] = N2d_ran_e1_s
		e2_ran_3d[k][i] = N2d_ran_e2_s
		e2kE_ran_3d[k][i] = e2kappaE_ran
		e2kB_ran_3d[k][i] = e2kappaB_ran



if zmean_type==1:
	np.savez(temp_dir_data+'k_3d_data_zmean.npz', e1=e1_3d, e2=e2_3d, e2kE=e2kE_3d, e2kB=e2kB_3d, e1_ran=e1_ran_3d, e2_ran=e2_ran_3d, \
			e2kE_ran=e2kE_ran_3d, e2kB_ran=e2kB_ran_3d)

# if nofz_type==1:
# 	np.savez(temp_dir_data+'k_3d_data_pofz.npz', e1=e1_3d, e2=e2_3d, e2kE=e2kE_3d, e2kB=e2kB_3d)

















