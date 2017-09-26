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
from numpy.linalg import inv

import config_data
for name in [name for name in dir(config_data) if not name.startswith("__")]:
    globals()[name] = getattr(config_data, name)

# load files
print('load data...')
mask_map = np.load(temp_dir_data+'mask_map.npz')['mask']
mask_bias = np.load(temp_dir_data+'mask_bias.npz')['mask']
jk_file = temp_dir_data+'jk_grid_data.npz'

# load correction factor ##########
if zmean_type==1:
	E1_file = np.loadtxt('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/temp_files/enrique_data_'+str(shear)+'_'+str(photoz)+'.txt')
	E2_file = np.loadtxt('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/temp_files/enrique_true.txt')
	factor = np.zeros((6,6))
	N = 0
	for i in range(6):
	    for j in range(i+1):
	        factor[i][j] = (E1_file[N][6]/E1_file[N][7]) #/(E2_file[N][6]/E2_file[N][7])
	        N = N+1
####################################

if zmean_type==1:
	kg_4d_data = np.load(temp_dir_data+'kg_4d_data_zmean.npz')
	k_3d_data = np.load(temp_dir_data+'k_3d_data_zmean.npz')

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
e2kE_ran = k_3d_data['e2kE_ran']
e2kB_ran = k_3d_data['e2kB_ran']
e1_ran = k_3d_data['e1_ran']
e2_ran = k_3d_data['e2_ran']

print('calculate zero-lag bias and full covariance...')

# print('e2kE')
# bias_e2kE = zerolag_bias(1, Nbin_z_l, Nbin_z_s, e2kE, kg, kg_ran, 'none', 'none', 'none', mask_bias, bias_type, factor)
# print(bias_e2kE)

# print('e2kB')
# bias_e2kB = zerolag_bias(1, Nbin_z_l, Nbin_z_s, e2kB, kg, kg_ran, 'none', 'none', 'none', mask_bias, bias_type, factor)
# print(bias_e2kB)

# print('k2eE')
# bias_k2eE = zerolag_bias(2, Nbin_z_l, Nbin_z_s, e1, k2e1, k2e1_ran, e2, k2e2, k2e2_ran, mask_bias, bias_type, factor)
# print(bias_k2eE)

# print('k2eB')
# bias_k2eB = zerolag_bias(2, Nbin_z_l, Nbin_z_s, -1*e2, k2e1, k2e1_ran, e1, k2e2, k2e2_ran, mask_bias, bias_type, factor)
# print(bias_k2eB)

print('e2kE')
Bias_e2kE, Bias_err_e2kE, bias_e2kE, bias_err_e2kE = covariance_bias(1, njk, Nbin_z_l, Nbin_z_s, e2kE, kg, kg_ran, 'none', 'none', 'none', mask_bias, jk_file, bias_type, factor)
print('e2kB')
Bias_e2kB, Bias_err_e2kB, bias_e2kB, bias_err_e2kB = covariance_bias(1, njk, Nbin_z_l, Nbin_z_s, e2kB, kg, kg_ran, 'none','none', 'none', mask_bias, jk_file, bias_type, factor)
print('k2eE')
Bias_k2eE, Bias_err_k2eE, bias_k2eE, bias_err_k2eE = covariance_bias(2, njk, Nbin_z_l, Nbin_z_s, e1, k2e1, k2e1_ran, e2, k2e2, k2e2_ran, mask_bias, jk_file, bias_type, factor)
print('k2eB')
Bias_k2eB, Bias_err_k2eB, bias_k2eB, bias_err_k2eB = covariance_bias(2, njk, Nbin_z_l, Nbin_z_s, -1*e2, k2e1, k2e1_ran, e1, k2e2, k2e2_ran, mask_bias, jk_file, bias_type, factor)

print('write out bias...')

np.savez(temp_dir_data+'mean_bias_'+str(shear)+'_'+str(photoz)+'.npz' ,\
Bias_e2kE=Bias_e2kE, Bias_err_e2kE=Bias_err_e2kE, Bias_e2kB=Bias_e2kB, Bias_err_e2kB=Bias_err_e2kB, \
Bias_k2eE=Bias_k2eE, Bias_err_k2eE=Bias_err_k2eE, Bias_k2eB=Bias_k2eB, Bias_err_k2eB=Bias_err_k2eB, \
bias_e2kE=bias_e2kE, bias_err_e2kE=bias_err_e2kE, bias_e2kB=bias_e2kB, bias_err_e2kB=bias_err_e2kB, \
bias_k2eE=bias_k2eE, bias_err_k2eE=bias_err_k2eE, bias_k2eB=bias_k2eB, bias_err_k2eB=bias_err_k2eB)


