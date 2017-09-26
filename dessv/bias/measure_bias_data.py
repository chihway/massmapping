"""
This script calculates the final redshift-dependent bias measurement 
from data. The final product is a 4D matrix, with a bias number (and 
error) for each lens-source redshift combination.

All combinations of gamma and kappa's are stored. 
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np 
import pylab as mplot
from bias_estimation import *

import config_data
for name in [name for name in dir(config_data) if not name.startswith("__")]:
    globals()[name] = getattr(config_data, name)

# load files
print('load data...')
mask_map = np.load(temp_dir_data+'mask_map.npz')['mask']
mask_bias = np.load(temp_dir_data+'mask_bias.npz')['mask']

if zmean_type==1:
	kg_4d_data = np.load(temp_dir_data+'kg_4d_data_zmean.npz')
	k_3d_data = np.load(temp_dir_data+'k_3d_data_zmean.npz')

	# load correction factor ##########
	E1_file = np.loadtxt('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/temp_files/enrique_data_im3shape_skynet.txt')
	E2_file = np.loadtxt('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/temp_files/enrique_true.txt')
	factor = np.zeros((6,6))
	N = 0
	for i in range(6):
	    for j in range(i+1):
	        factor[i][j] = (E1_file[N][6]/E1_file[N][7]) #/(E2_file[N][6]/E2_file[N][7])
	        N = N+1
####################################

if nofz_type==1:
	kg_4d_data = np.load(temp_dir_data+'kg_4d_data_pofz.npz')
	k_3d_data = np.load(temp_dir_data+'k_3d_data_pofz.npz')



k2e1 = kg_4d_data['k2e1']
k2e2 = kg_4d_data['k2e2']
kg = kg_4d_data['kg']
k2e1_ran = kg_4d_data['k2e1_ran']
k2e2_ran = kg_4d_data['k2e2_ran']
kg_ran = kg_4d_data['kg_ran']

e2kE = k_3d_data['e2kE']
e2kB = k_3d_data['e2kB']
e1 = k_3d_data['e1']
e2 = k_3d_data['e2']

Bias_e2kE = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_e2kE = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_e2kB = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_e2kB = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_k2e1 = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_k2e1 = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_k2e2 = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_k2e2 = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_k2e1B = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_k2e1B = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_k2e2B = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_k2e2B = np.zeros((Nbin_z_s, Nbin_z_l))

print('find cross-correlation between kg bins...')
# when we look at <kg k> for lens bin 2 and source bin 4
# there could also be cross correlation <kg k> between 
# lens bin 2 and contribution from lens bin 0 1 3 4 to 
# source bin 4, which needs to be removed 
kgkg_cross, kgkg_cross_mean, kgkg_cross_err, n_jk = jackknife_crosscorr(kg, mask_bias, jk_area, pix, temp_dir_data+'jk_grid_data.npz')
print(kgkg_cross)

M = np.load('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/photoz_corr/M_data.npz')['M']
print('find photo-z correction fcator...')
m, m_mean, m_err, n_jk = jackknife_m(kg, kg_ran, M, mask_bias, jk_area, pix, temp_dir_data+'jk_grid_data.npz')

print('calculate zero-lag bias...')
for i in range(Nbin_z_l):
	for j in range(Nbin_z_s):
		if j>=i:
			print('lens bin: '+str(i)+'; source bin: '+str(j))
			bias_e2kE = jackknife_bias(e2kE[j], kg[j][i], kg_ran[:,j,i], mask_bias, jk_area, pix, bias_type, temp_dir_data+'jk_grid_data.npz')
			bias_e2kB = jackknife_bias(e2kB[j], kg[j][i], kg_ran[:,j,i], mask_bias, jk_area, pix, bias_type, temp_dir_data+'jk_grid_data.npz')
			bias_k2e1 = jackknife_bias(e1[j], k2e1[j][i], k2e1_ran[:,j,i], mask_bias, jk_area, pix, bias_type, temp_dir_data+'jk_grid_data.npz')
			bias_k2e2 = jackknife_bias(e2[j], k2e2[j][i], k2e2_ran[:,j,i], mask_bias, jk_area, pix, bias_type, temp_dir_data+'jk_grid_data.npz')
			bias_k2e1B = jackknife_bias(-1*e2[j], k2e1[j][i], k2e1_ran[:,j,i], mask_bias, jk_area, pix, bias_type, temp_dir_data+'jk_grid_data.npz')
			bias_k2e2B = jackknife_bias(e1[j], k2e2[j][i], k2e2_ran[:,j,i], mask_bias, jk_area, pix, bias_type, temp_dir_data+'jk_grid_data.npz')
       
			Bias_e2kE[j][i] = bias_e2kE[0]*factor[j][i]
			Bias_err_e2kE[j][i] = bias_e2kE[4]*factor[j][i]
			Bias_e2kB[j][i] = bias_e2kB[0]*factor[j][i]
			Bias_err_e2kB[j][i] = bias_e2kB[4]*factor[j][i]

			Bias_k2e1[j][i] = bias_k2e1[0]*factor[j][i]
			Bias_err_k2e1[j][i] = bias_k2e1[4]*factor[j][i]
			Bias_k2e2[j][i] = bias_k2e2[0]*factor[j][i]
			Bias_err_k2e2[j][i] = bias_k2e2[4]*factor[j][i]

			Bias_k2e1B[j][i] = bias_k2e1B[0]*factor[j][i]
			Bias_err_k2e1B[j][i] = bias_k2e1B[4]*factor[j][i]
			Bias_k2e2B[j][i] = bias_k2e2B[0]*factor[j][i]
			Bias_err_k2e2B[j][i] = bias_k2e2B[4]*factor[j][i]

print(Bias_err_k2e2)

np.savez(temp_dir_data+'bias_data.npz', bias_e2kE=Bias_e2kE, bias_e2kB=Bias_e2kB, bias_k2e1=Bias_k2e1, bias_k2e2=Bias_k2e2, \
	bias_k2e1B=Bias_k2e1B, bias_k2e2B=Bias_k2e2B, \
	bias_err_e2kE=Bias_err_e2kE, bias_err_e2kB=Bias_err_e2kB, bias_err_k2e1=Bias_err_k2e1, bias_err_k2e2=Bias_err_k2e2, \
	bias_err_k2e1B=Bias_err_k2e1B, bias_err_k2e2B=Bias_err_k2e2B, \
	kgkg_cross=kgkg_cross, kgkg_cross_err=kgkg_cross_err, m=m, merr=m_err)


