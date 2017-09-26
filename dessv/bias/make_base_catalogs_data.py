"""
This is the first step in the analysis, which produces a 
set of catalogs to be used later. The catalogs include:

1) binned pofz catalogs
2) source (background) catalogs
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np 
import pyfits as pf
import os
import pylab as mplot
import healpy as hp
import sys
sys.path.append('../utils')
import utils

import config_data
for name in [name for name in dir(config_data) if not name.startswith("__")]:
	globals()[name] = getattr(config_data, name)

pz = 0

if pz==1:
	os.system('rm '+cat_gold_pofz_binned_name)

	# collapse pofz files #########

	print('making binned pofz files...')
	pdf_array = pf.open(cat_gold_pofz_name, mode='readonly', memmap=True)[1].data['ZPDF']
	z = np.arange(200)*1.0/200*1.8
	dz = z[1]-z[0]
	Z1 = np.arange(18)*0.1
	Z2 = (np.arange(18)+1)*0.1
	N = np.arange(200)
	Nbin = []

	for i in range(18):
		mask = (z>Z1[i])*(z<=Z2[i])
		Nbin.append(len(N[mask]))

	binned_pdf = np.zeros((len(pdf_array), 18))
	n = 0
	for i in range(18):
		binned_pdf[:,i] = np.sum(pdf_array[:,n:n+Nbin[i]], axis=1)*dz
		n += Nbin[i]

	c0 = pf.Column(name='ZPDF_BIN', format='18E', array=binned_pdf)
	CC = [c0]
	hdu = pf.new_table(CC, nrows=len(binned_pdf))
	hdu.writeto(cat_gold_pofz_binned_name, clobber=True)

# make shear files ##########

print('making shear files from '+str(shear)+' catalogs...')
tab1 = pf.open(cat_wl_info_name, mode='readonly', memmap=True)
tab2 = pf.open(cat_gold_pofz_binned_name, mode='readonly', memmap=True)
tab3 = pf.open(cat_gold_zmean_name, mode='readonly', memmap=True)

D = tab1[1].data
D2 = tab2[1].data
D3 = tab3[1].data
pofz = D2['ZPDF_BIN']
zmean = D3['ZMEAN']

if shear == 'im3shape':
	tab3 = pf.open(cat_wl_im3shape_name, mode='readonly', memmap=True)
	D3 = tab3[1].data
	mask = (D['sva1_spte_flags']==0)*(D['sva1_gold_flags']==0)*(D['sva1_gold_mag_flags']==0)*(D['im3shape_flags']==0)

if shear == 'ngmix':
	tab3 = pf.open(cat_wl_ngmix_name, mode='readonly', memmap=True)
	D3 = tab3[1].data
	mask = (D['sva1_spte_flags']==0)*(D['sva1_gold_flags']==0)*(D['sva1_gold_mag_flags']==0)*(D['ngmix_flags']==0)

for i in range(len(z_s)):
	print('source bin: '+str(z_s[i]))

	if nofz_type==1:
		zbin_mask = ((z_pofz_bin+0.05)>Z1_s[i])*((z_pofz_bin+0.05)<=Z2_s[i])
		n_weight = np.zeros(len(D[mask]))
		for j in range(len(z_pofz_bin[zbin_mask])):
		    n_weight += pofz[mask][:, int(z_pofz_bin[zbin_mask][j]/0.1)]
		nrows = len(D['RA'][mask])

		if shear == 'im3shape':
			
			# blinding!!! e1, e2 = unblind_arrays(D2['e1'][mask], D2['e2'][mask]*(-1))

			# nbc correction
			mean_nbc_m = np.mean(D3['nbc_m'][mask])

			c0 = pf.Column(name='ID', format='K', array=D['coadd_objects_id'][mask])
			c1 = pf.Column(name='RA', format='E', array=D['RA'][mask])
			c2 = pf.Column(name='DEC', format='E', array=D['DEC'][mask])
			c3 = pf.Column(name='S1_b', format='E', array=(D3['e1'][mask]-D3['nbc_c1'][mask])/(1.+mean_nbc_m))
			c4 = pf.Column(name='S2_b', format='E', array=(D3['e2'][mask]-D3['nbc_c2'][mask])/(1.+mean_nbc_m)*(-1))
			c5 = pf.Column(name='S1', format='E', array=(D3['e1'][mask]-D3['nbc_c1'][mask])/(1.+mean_nbc_m))
			c6 = pf.Column(name='S2', format='E', array=(D3['e2'][mask]-D3['nbc_c2'][mask])/(1.+mean_nbc_m)*(-1)) 
			c7 = pf.Column(name='W1', format='E', array=np.clip(D3['w'][mask], 0.0, 0.24**-2)) # clipping for v16 only
			c8 = pf.Column(name='W2', format='E', array=np.clip(D3['w'][mask], 0.0, 0.24**-2))
			c9 = pf.Column(name='SNR', format='E', array=D3['snr'][mask])
			c10 = pf.Column(name='PSF_E1', format='E', array=D3['mean_psf_e1_sky'][mask])
			c11 = pf.Column(name='PSF_E2', format='E', array=D3['mean_psf_e2_sky'][mask]*(-1))
			c12 = pf.Column(name='RGPP_RP', format='E', array=D3['mean_rgpp_rp'][mask])
			c13 = pf.Column(name='PSF_FWHM', format='E', array=D3['mean_psf_fwhm'][mask]) 
			c14 = pf.Column(name='NZ_weight', format='E', array=n_weight)

		if shear == 'ngmix':

			# blinding!!! e1, e2 = unblind_arrays(D2['e1'][mask], D2['e2'][mask]*(-1))
			
			# sensitivity correction
			sens1 = np.mean(D3['exp_e_sens_1'][mask])
			sens2 = np.mean(D3['exp_e_sens_2'][mask])

			c0 = pf.Column(name='ID', format='K', array=D['coadd_objects_id'][mask])
			c1 = pf.Column(name='RA', format='E', array=D['RA'][mask])
			c2 = pf.Column(name='DEC', format='E', array=D['DEC'][mask])
			c3 = pf.Column(name='S1_b', format='E', array=D3['exp_e_1'][mask]/sens1)
			c4 = pf.Column(name='S2_b', format='E', array=D3['exp_e_2'][mask]*(-1)/sens2)
			c5 = pf.Column(name='S1', format='E', array=D3['exp_e_1'][mask]/sens1)
			c6 = pf.Column(name='S2', format='E', array=D3['exp_e_2'][mask]*(-1)/sens2)
			c7 = pf.Column(name='W1', format='E', array=D3['exp_w'][mask])
			c8 = pf.Column(name='W2', format='E', array=D3['exp_w'][mask])
			c9 = pf.Column(name='SNR', format='E', array=D3['exp_s2n_w'][mask])
			c10 = pf.Column(name='PSF_E1', format='E', array=D3['psfrec_e_1'][mask])
			c11 = pf.Column(name='PSF_E2', format='E', array=D3['psfrec_e_2'][mask]*(-1))
			c12 = pf.Column(name='EXP_T_S2N', format='E', array=D3['exp_T_s2n'][mask])
			c13 = pf.Column(name='PSF_FWHM', format='E', array=D3['psfrec_T'][mask])
			c14 = pf.Column(name='NZ_weight', format='E', array=n_weight)

		CC = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14]
		hdu = pf.BinTableHDU.from_columns(CC, nrows=nrows)
		hdu.writeto(temp_dir_data+'background_'+str(shear)+'_tomo_pofz_'+str(photoz)+'_'+str(i)+'.fits', clobber=True)

	if zmean_type==1:
		zmask = mask*(zmean>Z1_s[i])*(zmean<=Z2_s[i])
		nrows = len(D['RA'][zmask])
		print(nrows)

		if shear == 'im3shape':
			
			# blinding!!! e1, e2 = unblind_arrays(D2['e1'][mask], D2['e2'][mask]*(-1))

			# nbc correction
			mean_nbc_m = np.mean(D3['nbc_m'][zmask])

			c0 = pf.Column(name='ID', format='K', array=D['coadd_objects_id'][zmask])
			c1 = pf.Column(name='RA', format='E', array=D['RA'][zmask])
			c2 = pf.Column(name='DEC', format='E', array=D['DEC'][zmask])
			c3 = pf.Column(name='S1_b', format='E', array=(D3['e1'][zmask]-D3['nbc_c1'][zmask])/(1.+mean_nbc_m))
			c4 = pf.Column(name='S2_b', format='E', array=(D3['e2'][zmask]-D3['nbc_c2'][zmask])/(1.+mean_nbc_m)*(-1))
			c5 = pf.Column(name='S1', format='E', array=(D3['e1'][zmask]-D3['nbc_c1'][zmask])/(1.+mean_nbc_m))
			c6 = pf.Column(name='S2', format='E', array=(D3['e2'][zmask]-D3['nbc_c2'][zmask])/(1.+mean_nbc_m)*(-1)) 
			c7 = pf.Column(name='W1', format='E', array=np.clip(D3['w'][zmask], 0.0, 0.24**-2)) # clipping for v16 only
			c8 = pf.Column(name='W2', format='E', array=np.clip(D3['w'][zmask], 0.0, 0.24**-2))
			c9 = pf.Column(name='SNR', format='E', array=D3['snr'][zmask])
			c10 = pf.Column(name='PSF_E1', format='E', array=D3['mean_psf_e1_sky'][zmask])
			c11 = pf.Column(name='PSF_E2', format='E', array=D3['mean_psf_e2_sky'][zmask]*(-1))
			c12 = pf.Column(name='RGPP_RP', format='E', array=D3['mean_rgpp_rp'][zmask])
			c13 = pf.Column(name='PSF_FWHM', format='E', array=D3['mean_psf_fwhm'][zmask]) 

		if shear == 'ngmix':

			# blinding!!! e1, e2 = unblind_arrays(D2['e1'][mask], D2['e2'][mask]*(-1))
			
			# sensitivity correction
			sens1 = np.mean(D3['exp_e_sens_1'][zmask])
			sens2 = np.mean(D3['exp_e_sens_2'][zmask])

			c0 = pf.Column(name='ID', format='K', array=D['coadd_objects_id'][zmask])
			c1 = pf.Column(name='RA', format='E', array=D['RA'][zmask])
			c2 = pf.Column(name='DEC', format='E', array=D['DEC'][zmask])
			c3 = pf.Column(name='S1_b', format='E', array=D3['exp_e_1'][zmask]/sens1)
			c4 = pf.Column(name='S2_b', format='E', array=D3['exp_e_2'][zmask]*(-1)/sens2)
			c5 = pf.Column(name='S1', format='E', array=D3['exp_e_1'][zmask]/sens1)
			c6 = pf.Column(name='S2', format='E', array=D3['exp_e_2'][zmask]*(-1)/sens2)
			c7 = pf.Column(name='W1', format='E', array=D3['exp_w'][zmask])
			c8 = pf.Column(name='W2', format='E', array=D3['exp_w'][zmask])
			c9 = pf.Column(name='SNR', format='E', array=D3['exp_s2n_w'][zmask])
			c10 = pf.Column(name='PSF_E1', format='E', array=D3['psfrec_e_1'][zmask])
			c11 = pf.Column(name='PSF_E2', format='E', array=D3['psfrec_e_2'][zmask]*(-1))
			c12 = pf.Column(name='EXP_T_S2N', format='E', array=D3['exp_T_s2n'][zmask])
			c13 = pf.Column(name='PSF_FWHM', format='E', array=D3['psfrec_T'][zmask])

		CC = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13]
		hdu = pf.BinTableHDU.from_columns(CC, nrows=nrows)
		hdu.writeto(temp_dir_data+'background_'+str(shear)+'_tomo_zmean_'+str(photoz)+'_'+str(i)+'.fits', clobber=True)


