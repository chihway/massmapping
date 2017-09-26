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

import config_sim
for name in [name for name in dir(config_sim) if not name.startswith("__")]:
    globals()[name] = getattr(config_sim, name)

# load files

print('load data...')
if ztrue_type==1:
	temp_name = ''
if zmean_type==1:		
	temp_name = '_zmean'
if nofz_type==1:
	temp_name = '_pofz'	


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
	        factor[i][j] = 1.0 #/(E_file[N][6]/E_file[N][7])
	        N = N+1

####################################

if mask_type==0:
	mask_mice_map = np.load(temp_dir_sim+'mask_mice_map.npz')['mask']
	mask_mice_bias = np.load(temp_dir_sim+'mask_mice_bias.npz')['mask']
	jk_file = temp_dir_sim+'jk_grid_sim.npz'
	kg_4d_data = np.load(temp_dir_sim+'kg_4d_sim'+str(temp_name)+'.npz')
	k_3d_data = np.load(temp_dir_sim+'k_3d_sim'+str(temp_name)+'.npz')

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

kk_ran = k_3d_data['k_ran']
g2kE_ran = k_3d_data['g2kE_ran']
g2kB_ran = k_3d_data['g2kB_ran']
g1_ran = k_3d_data['g1_ran']
g2_ran = k_3d_data['g2_ran']
e2kE_ran = k_3d_data['e2kE_ran']
e2kB_ran = k_3d_data['e2kB_ran']
e1_ran = k_3d_data['e1_ran']
e2_ran = k_3d_data['e2_ran']

# eg. For lens redshift bin 2 (0.4-0.6), we have 3 
# measurements of it, and the covariance (3x3) of 
# the three measurements from 20 jk samples  

Bias_k = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_k = np.zeros((Nbin_z_s, Nbin_z_l))

Bias_g2kE = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_g2kE = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_g2kB = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_g2kB = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_k2g1 = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_k2g1 = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_k2g2 = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_k2g2 = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_k2g1B = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_k2g1B = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_k2g2B = np.zeros((Nbin_z_s, Nbin_z_l))
Bias_err_k2g2B = np.zeros((Nbin_z_s, Nbin_z_l))

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

print('calculate zero-lag bias and full covariance...')
for i in range(Nbin_z_l):
	for j in range(Nbin_z_s):
		if j>=i:
			print('lens bin: '+str(i)+'; source bin: '+str(j))

			bias_k = jackknife_bias(kk[j], kk_ran[:,j], kg[j][i], kg_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)

			bias_g2kE = jackknife_bias(g2kE[j], g2kE_ran[:,j], kg[j][i], kg_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)
			bias_g2kB = jackknife_bias(g2kB[j], g2kB_ran[:,j], kg[j][i], kg_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)
			bias_k2g1 = jackknife_bias(g1[j], g1_ran[:,j], k2e1[j][i], k2e1_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)
			bias_k2g2 = jackknife_bias(g2[j], g2_ran[:,j], k2e2[j][i], k2e2_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)
			bias_k2g1B = jackknife_bias(-1*g2[j], -1*g2_ran[:,j], k2e1[j][i], k2e1_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)
			bias_k2g2B = jackknife_bias(g1[j], g1_ran[:,j], k2e2[j][i], k2e2_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)


			bias_e2kE = jackknife_bias(e2kE[j], e2kE_ran[:,j], kg[j][i], kg_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)
			bias_e2kB = jackknife_bias(e2kB[j], e2kB_ran[:,j], kg[j][i], kg_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)
			bias_k2e1 = jackknife_bias(e1[j], e1_ran[:,j], k2e1[j][i], k2e1_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)
			bias_k2e2 = jackknife_bias(e2[j], e2_ran[:,j], k2e2[j][i], k2e2_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)
			bias_k2e1B = jackknife_bias(-1*e2[j], -1*e2_ran[:,j], k2e1[j][i], k2e1_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)
			bias_k2e2B = jackknife_bias(e1[j], e1_ran[:,j], k2e2[j][i], k2e2_ran[:,j,i], mask_mice_bias, jk_area, pix, bias_type, jk_file)
       
			Bias_k[j][i] = bias_k[0]*factor[j][i]
			Bias_err_k[j][i] = bias_k[4]*factor[j][i]

			Bias_g2kE[j][i] = bias_g2kE[0]*factor[j][i]
			Bias_err_g2kE[j][i] = bias_g2kE[4]*factor[j][i]
			Bias_g2kB[j][i] = bias_g2kB[0]*factor[j][i]
			Bias_err_g2kB[j][i] = bias_g2kB[4]*factor[j][i]

			Bias_k2g1[j][i] = bias_k2g1[0]*factor[j][i]
			Bias_err_k2g1[j][i] = bias_k2g1[4]*factor[j][i]
			Bias_k2g2[j][i] = bias_k2g2[0]*factor[j][i]
			Bias_err_k2g2[j][i] = bias_k2g2[4]*factor[j][i]
			Bias_k2g1B[j][i] = bias_k2g1B[0]*factor[j][i]
			Bias_err_k2g1B[j][i] = bias_k2g1B[4]*factor[j][i]
			Bias_k2g2B[j][i] = bias_k2g2B[0]*factor[j][i]
			Bias_err_k2g2B[j][i] = bias_k2g2B[4]*factor[j][i]

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

if mask_type==0:
	temp_name = 'bias_sim_'+str(shear)+str(temp_name)+'.npz'
if mask_type==1:
	temp_name = 'bias_sim_sv_'+str(shear)+str(temp_name)+'.npz'	

np.savez(temp_dir_sim+temp_name, \
	bias_k=Bias_k, bias_err_k=Bias_err_k, \
	bias_g2kE=Bias_g2kE, bias_g2kB=Bias_g2kB, bias_k2g1=Bias_k2g1, bias_k2g2=Bias_k2g2, bias_k2g1B=Bias_k2g1B, bias_k2g2B=Bias_k2g2B, \
	bias_err_g2kE=Bias_err_g2kE, bias_err_g2kB=Bias_err_g2kB, bias_err_k2g1=Bias_err_k2g1, bias_err_k2g2=Bias_err_k2g2, bias_err_k2g1B=Bias_err_k2g1B, bias_err_k2g2B=Bias_err_k2g2B, \
	bias_e2kE=Bias_e2kE, bias_e2kB=Bias_e2kB, bias_k2e1=Bias_k2e1, bias_k2e2=Bias_k2e2, bias_k2e1B=Bias_k2e1B, bias_k2e2B=Bias_k2e2B,\
	bias_err_e2kE=Bias_err_e2kE, bias_err_e2kB=Bias_err_e2kB, bias_err_k2e1=Bias_err_k2e1, bias_err_k2e2=Bias_err_k2e2, bias_err_k2e1B=Bias_err_k2e1B, bias_err_k2e2B=Bias_err_k2e2B)


