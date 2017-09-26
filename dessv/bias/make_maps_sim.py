"""
This script runs through and makes all maps needed for the 
bias calculation for sims. The final product is a 4D matrix for 
kappa_g and gamma_g, 3D for kappa and gamma.

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
import pylab as mplot

import config_sim
for name in [name for name in dir(config_sim) if not name.startswith("__")]:
	globals()[name] = getattr(config_sim, name)

random.seed(ran_seed)

# load data and mask #######

print('load data...')
if mask_type==0:
	mask_mice_map = np.load(temp_dir_sim+'mask_mice_map.npz')['mask']
	mask_mice_bias = np.load(temp_dir_sim+'mask_mice_bias.npz')['mask']

if mask_type==1:
	mask_mice_map = np.load(temp_dir_sim+'mask_mice_sv_map_'+str(mask_flag)+'.npz')['mask']
	mask_mice_bias = np.load(temp_dir_sim+'mask_mice_sv_bias_'+str(mask_flag)+'.npz')['mask']


D = pf.open(temp_dir_sim+'foreground_mice_sv.fits')[1].data

ra = D['RA']
dec = D['DEC']

if ztrue_type==1:
	z = D['Z_true']
# 	nofz_l = np.load(temp_dir_sim+'nofz_sims_lens_true.npz')['nofz']
# 	nofz_s = np.load(temp_dir_sim+'nofz_sims_source_true.npz')['nofz'] 

if zmean_type==1:
	z = D['Z_mean']
# 	nofz_l = np.load(temp_dir_sim+'nofz_sims_lens_gaussian_0.03.npz')['nofz']
# 	nofz_s = np.load(temp_dir_sim+'nofz_sims_source_gaussian_0.03.npz')['nofz'] 


ran = pf.open(ran_name)[1].data
ran_len = len(ran)

# make kappa_g/gamma_g maps ########

print('make kappa_g/gamma_g maps...')
ra_p = ra_ref + (ra-ra_ref)*np.cos(dec/180.*np.pi)
g_3d_l = np.zeros((Nbin_z_l, Nbin_dec, Nbin_ra))
kg_4d = np.zeros((Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))
kg_ran_4d = np.zeros((N_ran,Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))
k2e1_4d = np.zeros((Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))
k2e2_4d = np.zeros((Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))
k2e1_ran_4d = np.zeros((N_ran, Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))
k2e2_ran_4d = np.zeros((N_ran, Nbin_z_s, Nbin_z_l, Nbin_dec, Nbin_ra))

# first loop through lens bins
for i in range(Nbin_z_l):
	print('lens bin'+str(i))
	mask = (z>Z1_l[i])*(z<=Z2_l[i])

	# then loop through source bins
	for j in range(Nbin_z_s):

		# get mean redshift in bin?
		if zmean_type==1:
			D2 = pf.open(temp_dir_sim+'background_mice_sv_'+str(shear)+'_zmean_'+str(photoz)+'_'+str(j)+'.fits')[1].data
			zz = D2['Z_mean']
			zs_mean = np.mean(zz)
		if ztrue_type==1:
			D2 = pf.open(temp_dir_sim+'background_mice_sv_'+str(shear)+'_'+str(photoz)+'_'+str(j)+'.fits')[1].data
			zz = D2['Z_true']
			zs_mean = np.mean(zz)
		

		print('source bin'+str(j))
		if j>=i:
			N2d_bin, edges = np.histogramdd(np.array([dec[mask], ra_p[mask]]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)))
			g_3d_l[i][mask_mice_map==1] = (N2d_bin[mask_mice_map==1] - np.mean(N2d_bin[mask_mice_map==1]))/np.mean(N2d_bin[mask_mice_map==1])
			#kg_4d[j][i] = kappag_map_bin_nofz(N2d_bin, mask_mice_map, nofz_l[i], nofz_s[j], cosmo, smooth_kernal, smooth_scale)
			#kg_4d[j][i] = kappag_map_bin_nofz2(N2d_bin, mask_mice_map, Z1_l[i], Z2_l[i], nofz_s[j], cosmo, smooth_kernal, smooth_scale)
			kg_4d[j][i] = kappag_map_bin(N2d_bin, mask_mice_map, None, Z1_l[i], Z2_l[i], z_s[j], cosmo, smooth_kernal, smooth_scale)
			k2e1, k2e2 = k2g_fft(kg_4d[j][i], kg_4d[j][i]*0.0, pix, pad=True)
			k2e1_4d[j][i] = k2e1
			k2e2_4d[j][i] = k2e2			

			# make the random maps
			Ngal = len(dec[mask])
			for k in range(N_ran):
				print(k)
				ran_ids = random.choice(range(ran_len), size=Ngal, replace=False)
				ra_ran = ran['RA'][ran_ids]
				dec_ran = ran['DEC'][ran_ids]
				ra_ran_p = ra_ref + (ra_ran-ra_ref)*np.cos(dec_ran/180.*np.pi)
				N2d_bin, edges = np.histogramdd(np.array([dec_ran, ra_ran_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)))
				#kg_ran_4d[k][j][i] = kappag_map_bin_nofz(N2d_bin, mask_mice_map, nofz_l[i], nofz_s[j], cosmo, smooth_kernal, smooth_scale)
				#kg_ran_4d[k][j][i] = kappag_map_bin_nofz2(N2d_bin, mask_mice_map, Z1_l[i], Z2_l[i], nofz_s[j], cosmo, smooth_kernal, smooth_scale)
				kg_ran_4d[k][j][i] = kappag_map_bin(N2d_bin, mask_mice_map, None, Z1_l[i], Z2_l[i], z_s[j], cosmo, smooth_kernal, smooth_scale)
				k2e1, k2e2 = k2g_fft(kg_ran_4d[k][j][i], kg_ran_4d[k][j][i]*0.0, pix, pad=True)
				k2e1_ran_4d[k][j][i] = k2e1
				k2e2_ran_4d[k][j][i] = k2e2	

if mask_type==0:
	if ztrue_type==1:
		np.savez(temp_dir_sim+'kg_4d_sim.npz', kg=kg_4d, kg_ran=kg_ran_4d, k2e1=k2e1_4d, k2e2=k2e2_4d, k2e1_ran=k2e1_ran_4d, k2e2_ran=k2e2_ran_4d, g_3d_l=g_3d_l)
	if zmean_type==1:
		np.savez(temp_dir_sim+'kg_4d_sim_zmean.npz', kg=kg_4d, kg_ran=kg_ran_4d, k2e1=k2e1_4d, k2e2=k2e2_4d, k2e1_ran=k2e1_ran_4d, k2e2_ran=k2e2_ran_4d, g_3d_l=g_3d_l)

if mask_type==1:
	if ztrue_type==1:
		np.savez(temp_dir_sim+'kg_4d_sim_sv_'+str(mask_flag)+'.npz', kg=kg_4d, kg_ran=kg_ran_4d, k2e1=k2e1_4d, k2e2=k2e2_4d, k2e1_ran=k2e1_ran_4d, k2e2_ran=k2e2_ran_4d, g_3d_l=g_3d_l)
	if zmean_type==1:
		np.savez(temp_dir_sim+'kg_4d_sim_sv_zmean_'+str(mask_flag)+'.npz', kg=kg_4d, kg_ran=kg_ran_4d, k2e1=k2e1_4d, k2e2=k2e2_4d, k2e1_ran=k2e1_ran_4d, k2e2_ran=k2e2_ran_4d, g_3d_l=g_3d_l)

# make kappa/gamma maps ########

print('make kappa/gamma maps...')
g_3d_s = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))

k_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))
g1_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))
g2_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))
g2kE_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))
g2kB_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))
e1_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))
e2_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))
e2kE_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))
e2kB_3d = np.zeros((Nbin_z_s, Nbin_dec, Nbin_ra))

for i in range(Nbin_z_s):
	print('source bin'+str(i))

	if ztrue_type==1:
		D = pf.open(temp_dir_sim+'background_mice_sv_'+str(shear)+'_'+str(photoz)+'_'+str(i)+'.fits')[1].data
	if zmean_type==1:
		D = pf.open(temp_dir_sim+'background_mice_sv_'+str(shear)+'_zmean_'+str(photoz)+'_'+str(i)+'.fits')[1].data
	
	ra = D['RA']
	dec = D['DEC']
	g1 = D['S1']
	g2 = D['S2']
	e1 = D['E1']
	e2 = D['E2']
	kk = D['KAPPA']
	
	# make a few bootstrap simple random --> replace this with function

	ra_p = ra_ref + (ra-ra_ref)*np.cos(dec/180.*np.pi)

	N2d_g1, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=g1)
	N2d_g2, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=g2)
	N2d_e1, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=e1)
	N2d_e2, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=e2)
	N2d_k, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)), weights=kk)
	N2d, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)))

	N2d_patch = N2d*1.0
	N2d_patch[N2d_patch==0]=1.0
	N2d_g1 /= N2d_patch
	N2d_g2 /= N2d_patch
	N2d_e1 /= N2d_patch
	N2d_e2 /= N2d_patch	
	N2d_k /= N2d_patch
	
	N2d_s = N2d*1.0
	N2d_g1_s = N2d_g1*1.0 
	N2d_g2_s = N2d_g2*1.0 
	N2d_e1_s = N2d_e1*1.0 
	N2d_e2_s = N2d_e2*1.0 
	N2d_k_s = N2d_k*1.0 
	
	if (smooth>0.0):
		N2d_s = smooth_map(N2d_s, mask_mice_map, smooth_kernal, smooth_scale, 0.0) 
		N2d_g1_s = smooth_map(N2d_g1_s, mask_mice_map, smooth_kernal, smooth_scale, 0.0) 
		N2d_g2_s = smooth_map(N2d_g2_s, mask_mice_map, smooth_kernal, smooth_scale, 0.0) 
		N2d_e1_s = smooth_map(N2d_e1_s, mask_mice_map, smooth_kernal, smooth_scale, 0.0) 
		N2d_e2_s = smooth_map(N2d_e2_s, mask_mice_map, smooth_kernal, smooth_scale, 0.0) 
		N2d_k_s = smooth_map(N2d_k_s, mask_mice_map, smooth_kernal, smooth_scale, 0.0) 

	# conversion
	g2kappaE, g2kappaB = g2k_fft(N2d_g1_s, N2d_g2_s, pix, pad=True)
	e2kappaE, e2kappaB = g2k_fft(N2d_e1_s, N2d_e2_s, pix, pad=True)
	
	g_3d_s[i][mask_mice_map==1] = (N2d_s[mask_mice_map==1] - np.mean(N2d_s[mask_mice_map==1]))/np.mean(N2d_s[mask_mice_map==1])
	k_3d[i] = N2d_k_s
	g1_3d[i] = N2d_g1_s
	g2_3d[i] = N2d_g2_s
	g2kE_3d[i] = g2kappaE
	g2kB_3d[i] = g2kappaB
	e1_3d[i] = N2d_e1_s
	e2_3d[i] = N2d_e2_s
	e2kE_3d[i] = e2kappaE
	e2kB_3d[i] = e2kappaB

if mask_type==0:
	if ztrue_type==1:
		temp_name = 'k_3d_sim.npz'
	if zmean_type==1:
		temp_name = 'k_3d_sim_zmean.npz'
if mask_type==1:
	if ztrue_type==1:
		temp_name = 'k_3d_sim_sv_'+str(mask_flag)+'.npz'
	if zmean_type==1:
		temp_name = 'k_3d_sim_sv_zmean_'+str(mask_flag)+'.npz'

np.savez(temp_dir_sim+temp_name, k=k_3d, g1=g1_3d, g2=g2_3d, g2kE=g2kE_3d, g2kB=g2kB_3d, e1=e1_3d, e2=e2_3d, e2kE=e2kE_3d, e2kB=e2kB_3d) 






