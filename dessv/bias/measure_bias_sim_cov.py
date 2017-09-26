"""
This script calculates the final redshift-dependent bias measurement 
from sims. The final product is a 4D matrix, with a bias number (and 
error) for each lens-source redshift combination.

All combinations of gamma and kappa's are stored. 
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np 
import pylab as mplot
from bias_estimation import *
from numpy.linalg import inv

import config_sim
for name in [name for name in dir(config_sim) if not name.startswith("__")]:
    globals()[name] = getattr(config_sim, name)


# load correction factor ##########
if zmean_type==1:
	E1_file = np.loadtxt('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/temp_files/enrique_gaussian_003.txt')
	E2_file = np.loadtxt('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/temp_files/enrique_true.txt')
	factor = np.zeros((6,6))
	N = 0
	for i in range(6):
	    for j in range(i+1):
	        factor[i][j] = (E1_file[N][6]/E1_file[N][7]) #/(E2_file[N][6]/E2_file[N][7])
	        N = N+1

if ztrue_type==1:
	E_file = np.loadtxt('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/temp_files/enrique_true.txt')
	factor = np.zeros((6,6))
	N = 0
	for i in range(6):
	    for j in range(i+1):
	        factor[i][j] = (E_file[N][6]/E_file[N][7])
	        N = N+1
####################################

# load files

print('load data...')
if ztrue_type==1:
	temp_name = ''
if zmean_type==1:		
	temp_name = '_zmean'
if nofz_type==1:
	temp_name = '_pofz'	

if mask_type==0:
	mask_mice_map = np.load(temp_dir_sim+'mask_mice_map.npz')['mask']
	mask_mice_bias = np.load(temp_dir_sim+'mask_mice_bias.npz')['mask']
	jk_file = temp_dir_sim+'jk_grid_sim.npz'
	kg_4d_data = np.load(temp_dir_sim+'kg_4d_sim'+str(temp_name)+'.npz')
	k_3d_data = np.load(temp_dir_sim+'k_3d_sim'+str(temp_name)+'.npz')
	njk = 20

if mask_type==1:
	mask_mice_map = np.load(temp_dir_sim+'mask_mice_sv_map_'+str(mask_flag)+'.npz')['mask']
	mask_mice_bias = np.load(temp_dir_sim+'mask_mice_sv_bias_'+str(mask_flag)+'.npz')['mask']
	jk_file = temp_dir_sim+'jk_grid_sim_sv_'+str(mask_flag)+'.npz'
	kg_4d_data = np.load(temp_dir_sim+'kg_4d_sim_sv'+str(temp_name)+'_'+str(mask_flag)+'.npz')
	k_3d_data = np.load(temp_dir_sim+'k_3d_sim_sv'+str(temp_name)+'_'+str(mask_flag)+'.npz')

k2e1 = kg_4d_data['k2e1']
k2e2 = kg_4d_data['k2e2']
kg = kg_4d_data['kg']
k2e1_ran = kg_4d_data['k2e1_ran']
k2e2_ran = kg_4d_data['k2e2_ran']
kg_ran = kg_4d_data['kg_ran']

kk = k_3d_data['k']
g2kE = k_3d_data['g2kE']
g2kB = k_3d_data['g2kB']
g1 = k_3d_data['g1']
g2 = k_3d_data['g2']
e2kE = k_3d_data['e2kE']
e2kB = k_3d_data['e2kB']
e1 = k_3d_data['e1']
e2 = k_3d_data['e2']


print('calculate zero-lag bias and full covariance...')

# # here's NOT using the JK errors
# print('k')
# bias_k = zerolag_bias(1, Nbin_z_l, Nbin_z_s, kk, kg, kg_ran, 'none', 'none', 'none', mask_mice_bias, bias_type, factor)
# print('g2kE')
# bias_g2kE = zerolag_bias(1, Nbin_z_l, Nbin_z_s, g2kE, kg, kg_ran, 'none', 'none', 'none', mask_mice_bias, bias_type, factor)
# print('g2kB')
# bias_g2kB = zerolag_bias(1, Nbin_z_l, Nbin_z_s, g2kB, kg, kg_ran, 'none', 'none', 'none', mask_mice_bias, bias_type, factor)
# print('k2gE')
# bias_k2gE = zerolag_bias(2, Nbin_z_l, Nbin_z_s, g1, k2e1, k2e1_ran, g2, k2e2, k2e2_ran, mask_mice_bias, bias_type, factor)
# print('k2gB')
# bias_k2gB = zerolag_bias(2, Nbin_z_l, Nbin_z_s, -1*g2, k2e1, k2e1_ran, g1, k2e2, k2e2_ran, mask_mice_bias, bias_type, factor)
# print('e2kE')
# bias_e2kE = zerolag_bias(1, Nbin_z_l, Nbin_z_s, e2kE, kg, kg_ran, 'none', 'none', 'none', mask_mice_bias, bias_type, factor)
# print('e2kB')
# bias_e2kB = zerolag_bias(1, Nbin_z_l, Nbin_z_s, e2kB, kg, kg_ran, 'none', 'none', 'none', mask_mice_bias, bias_type, factor)
# print('k2eE')
# bias_k2eE = zerolag_bias(2, Nbin_z_l, Nbin_z_s, e1, k2e1, k2e1_ran, e2, k2e2, k2e2_ran, mask_mice_bias, bias_type, factor)
# print('k2eB')
# bias_k2eB = zerolag_bias(2, Nbin_z_l, Nbin_z_s, -1*e2, k2e1, k2e1_ran, e1,  k2e2, k2e2_ran, mask_mice_bias, bias_type, factor)

# print('write out bias...')

# if mask_type==0:
# 	temp_sv_name = ''
# if mask_type==1:
# 	temp_sv_name = '_sv'

# np.savez(temp_dir_sim+'bias_cov_'+str(shear)+str(temp_sv_name)+str(temp_name)+'.npz', \
# bias_k=bias_k, bias_g2kE=bias_g2kE, bias_g2kB=bias_g2kB, bias_k2gE=bias_k2gE, bias_k2gB=bias_k2gB, \
# bias_e2kE=bias_e2kE, bias_e2kB=bias_e2kB, bias_k2eE=bias_k2eE, bias_k2eB=bias_k2eB)



print('k')
Bias_k, Bias_err_k, bias_k, bias_err_k = covariance_bias(1, njk, Nbin_z_l, Nbin_z_s, kk, kg, kg_ran, 'none', 'none', 'none', mask_mice_bias, jk_file, bias_type, factor)
print('g2kE')
Bias_g2kE, Bias_err_g2kE, bias_g2kE, bias_err_g2kE = covariance_bias(1, njk, Nbin_z_l, Nbin_z_s, g2kE, kg, kg_ran, 'none', 'none', 'none', mask_mice_bias, jk_file, bias_type, factor)
print('g2kB')
Bias_g2kB, Bias_err_g2kB, bias_g2kB, bias_err_g2kB = covariance_bias(1, njk, Nbin_z_l, Nbin_z_s, g2kB, kg, kg_ran, 'none', 'none', 'none', mask_mice_bias, jk_file, bias_type, factor)
print('k2gE')
Bias_k2gE, Bias_err_k2gE, bias_k2gE, bias_err_k2gE = covariance_bias(2, njk, Nbin_z_l, Nbin_z_s, g1, k2e1, k2e1_ran, g2, k2e2, k2e2_ran, mask_mice_bias, jk_file, bias_type, factor)
print('k2gB')
Bias_k2gB, Bias_err_k2gB, bias_k2gB, bias_err_k2gB = covariance_bias(2, njk, Nbin_z_l, Nbin_z_s, -1*g2, k2e1, k2e1_ran, g1, k2e2, k2e2_ran, mask_mice_bias, jk_file, bias_type, factor)
print('e2kE')
Bias_e2kE, Bias_err_e2kE, bias_e2kE, bias_err_e2kE = covariance_bias(1, njk, Nbin_z_l, Nbin_z_s, e2kE, kg, kg_ran, 'none', 'none', 'none', mask_mice_bias, jk_file, bias_type, factor)
print('e2kB')
Bias_e2kB, Bias_err_e2kB, bias_e2kB, bias_err_e2kB = covariance_bias(1, njk, Nbin_z_l, Nbin_z_s, e2kB, kg, kg_ran, 'none', 'none', 'none', mask_mice_bias, jk_file, bias_type, factor)
print('k2eE')
Bias_k2eE, Bias_err_k2eE, bias_k2eE, bias_err_k2eE = covariance_bias(2, njk, Nbin_z_l, Nbin_z_s, e1, k2e1, k2e1_ran, e2, k2e2, k2e2_ran, mask_mice_bias, jk_file, bias_type, factor)
print('k2eB')
Bias_k2eB, Bias_err_k2eB, bias_k2eB, bias_err_k2eB = covariance_bias(2, njk, Nbin_z_l, Nbin_z_s, -1*e2, k2e1, k2e1_ran, e1, k2e2, k2e2_ran, mask_mice_bias, jk_file, bias_type, factor)

print('write out bias...')

if mask_type==0:
	temp_sv_name = ''
if mask_type==1:
	temp_sv_name = '_sv'


# saving the "inverse bias"
np.savez(temp_dir_sim+'mean_bias_cov_'+str(shear)+str(temp_sv_name)+str(temp_name)+'.npz', \
Bias_k=Bias_k, Bias_err_k=Bias_err_k, \
Bias_g2kE=Bias_g2kE, Bias_err_g2kE=Bias_err_g2kE, Bias_g2kB=Bias_g2kB, Bias_err_g2kB=Bias_err_g2kB, \
Bias_k2gE=Bias_k2gE, Bias_err_k2gE=Bias_err_k2gE, Bias_k2gB=Bias_k2gB, Bias_err_k2gB=Bias_err_k2gB, \
Bias_e2kE=Bias_e2kE, Bias_err_e2kE=Bias_err_e2kE, Bias_e2kB=Bias_e2kB, Bias_err_e2kB=Bias_err_e2kB, \
Bias_k2eE=Bias_k2eE, Bias_err_k2eE=Bias_err_k2eE, Bias_k2eB=Bias_k2eB, Bias_err_k2eB=Bias_err_k2eB, 
bias_k=bias_k, bias_err_k=bias_err_k, \
bias_g2kE=bias_g2kE, bias_err_g2kE=bias_err_g2kE, bias_g2kB=bias_g2kB, bias_err_g2kB=bias_err_g2kB, \
bias_k2gE=bias_k2gE, bias_err_k2gE=bias_err_k2gE, bias_k2gB=bias_k2gB, bias_err_k2gB=bias_err_k2gB, \
bias_e2kE=bias_e2kE, bias_err_e2kE=bias_err_e2kE, bias_e2kB=bias_e2kB, bias_err_e2kB=bias_err_e2kB, \
bias_k2eE=bias_k2eE, bias_err_k2eE=bias_err_k2eE, bias_k2eB=bias_k2eB, bias_err_k2eB=bias_err_k2eB)




