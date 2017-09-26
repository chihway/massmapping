"""
This is the first step in the analysis for sims, which 
produces a set of catalogs to be used later. The catalogs 
include:

1) source (background) catalogs with matching number counts
and shape noise 
2) lens (foreground) catalogs
3) source catalogs with pofz 
4) lens catalogs with pofz
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np 
import pyfits as pf
import sys, os
import pylab as mplot
from numpy import random
sys.path.append('../utils')
from utils import *

import config_sim
for name in [name for name in dir(config_sim) if not name.startswith("__")]:
	globals()[name] = getattr(config_sim, name)

random.seed(ran_seed)

# remove all old files #########

# print('removing old files...')
# os.system('rm '+temp_dir_sim+'background_mice_sv_'+str(shear)+'_'+str(photoz)+'_*.fits')
# os.system('rm '+temp_dir_sim+'foreground_mice_sv.fits')

# load MICE #########

print('load MICE...')
tab1 = pf.open(cat_mice_coord_name, mode='readonly', memmap=True)
tab2 = pf.open(cat_mice_shear_name, mode='readonly', memmap=True)
D1 = tab1[1].data
D2 = tab2[1].data
RA = D1['RA']
DEC = D1['DEC']
G1 = D2['GAMMA1']
G2 = D2['GAMMA2']
Z = D1['Z']
K = D2['KAPPA']
MAG = D1['MAG_I']
MAG = MAG - 0.8 * (np.arctan(1.5*Z) - 0.1489) # add redshift evolution

zmean_s = pf.open(cat_mice_pz_source_name)[1].data['ZMEAN']
zmean_l = pf.open(cat_mice_pz_lens_name)[1].data['ZMEAN']

# make shear files ##########

print('making shear catalogs...')

if ztrue_type==1 or zmean_type==1:
	for i in range(Nbin_z_s):
		
		# data = pf.open(temp_dir_data+'background_'+str(shear)+'_tomo_pofz_'+str(photoz)+'_'+str(i)+'.fits')[1].data
		data = pf.open(temp_dir_data+'background_'+str(shear)+'_tomo_zmean_'+str(photoz)+'_'+str(i)+'.fits')[1].data

		s1 = data['S1']
		s2 = data['S1'] 
		w1 = data['W1']
		w2 = data['W2']
		#wn = data['NZ_weight']

		# first true redshift
		# calculate number counts: sum of NZ_weight in that bin
		print('galaxy per arcmin^2 for source bin '+str(i)+':', len(s1)/139./60/60)
		print('galaxies to draw from MICE:', int(len(s1)/139.*(decmax-decmin)*(ramax-ramin)*np.cos((decmin+decmax)/2/180.*np.pi)))
		Ngal = int(len(s1)/139.*(decmax-decmin)*(ramax-ramin)*np.cos((decmin+decmax)/2/180.*np.pi))

		# calculate shape noise: draw from PDF of NZ_weight-weighted, w-weighted 
		# histogram, new entries have neither weight nor nz_weight 
		NN = mplot.hist(s1, weights=w1, histtype='step', bins=200, range=(-1,1), normed=1)
		e1_ran = get_random_from_hist(NN, Ngal, bin=500)
		NN = mplot.hist(s2, weights=w2, histtype='step', bins=200, range=(-1,1), normed=1)
		e2_ran = get_random_from_hist(NN, Ngal, bin=500)

		# test photo-z
		if ztrue_type==1:
			mask = (RA<ramax)*(RA>ramin)*(DEC<decmax)*(DEC>decmin)*(Z>Z1_s[i])*(Z<=Z2_s[i]) 
		
		if zmean_type==1:
			mask = (RA<ramax)*(RA>ramin)*(DEC<decmax)*(DEC>decmin)*(zmean_s>Z1_s[i])*(zmean_s<=Z2_s[i]) 
		
		nrows = len(RA[mask])
		random.seed(seed+100)
		print(nrows, Ngal)
		if nrows>Ngal:
			ids = random.choice(range(nrows), size=Ngal, replace=False)
		else: 
			ids = range(nrows)
			e1_ran = e1_ran[ids]
			e2_ran = e2_ran[ids]

		c1 = pf.Column(name='RA', format='E', array=RA[mask][ids])
		c2 = pf.Column(name='DEC', format='E', array=DEC[mask][ids])
		c3 = pf.Column(name='S1', format='E', array=G1[mask][ids]*(-1))
		c4 = pf.Column(name='S2', format='E', array=G2[mask][ids]*(-1))
		c5 = pf.Column(name='E1', format='E', array=G1[mask][ids]*(-1)+e1_ran)
		c6 = pf.Column(name='E2', format='E', array=G2[mask][ids]*(-1)+e2_ran)
		c7 = pf.Column(name='Z_true', format='E', array=Z[mask][ids])
		c8 = pf.Column(name='Z_mean', format='E', array=zmean_s[mask][ids])
		c9 = pf.Column(name='KAPPA', format='E', array=K[mask][ids])

		CC = [c1, c2, c3, c4, c5, c6, c7, c8, c9]
		hdu = pf.new_table(CC, nrows=len(Z[mask][ids]))
		if ztrue_type==1:
			hdu.writeto(temp_dir_sim+'background_mice_sv_'+str(shear)+'_'+str(photoz)+'_'+str(i)+'.fits', clobber=True)
		if zmean_type==1:
			hdu.writeto(temp_dir_sim+'background_mice_sv_'+str(shear)+'_zmean_'+str(photoz)+'_'+str(i)+'.fits', clobber=True)


if nofz_type==1:

	# next nofz for source
	data = pf.open(temp_dir_data+'background_'+str(shear)+'_tomo_pofz_'+str(photoz)+'_0.fits')[1].data
	s1 = data['S1']
	s2 = data['S1'] 
	w1 = data['W1']
	w2 = data['W2']

	print('galaxy per arcmin^2:', len(w1)/139./60/60)
	print('galaxies to draw from MICE:', int(len(w1)/139.*(decmax-decmin)*(ramax-ramin)*np.cos((decmin+decmax)/2/180.*np.pi)))
	Ngal = int(len(w1)*1.0/139.*(decmax-decmin)*(ramax-ramin)*np.cos((decmin+decmax)/2/180.*np.pi))

	# calculate shape noise: draw from PDF of NZ_weight-weighted, w-weighted 
	# histogram, new entries have neither weight nor nz_weight 
	NN = mplot.hist(s1, weights=w1, histtype='step', bins=200, range=(-1,1), normed=1)
	e1_ran = get_random_from_hist(NN, Ngal, bin=500)
	NN = mplot.hist(s2, weights=w2, histtype='step', bins=200, range=(-1,1), normed=1)
	e2_ran = get_random_from_hist(NN, Ngal, bin=500)

	mask = (RA<ramax)*(RA>ramin)*(DEC<decmax)*(DEC>decmin)
	nrows = len(RA[mask])
	random.seed(seed)
	ids = random.choice(range(nrows), size=Ngal, replace=False)

	c0 = pf.Column(name='ZPDF_BIN', format='18E', array=pofz_s[mask][ids]) 
	c1 = pf.Column(name='Z_true', format='E', array=Z[mask][ids]) 
	CC = [c0, c1]
	hdu = pf.new_table(CC, nrows=Ngal)
	hdu.writeto(temp_dir_sim+'background_mice_sv_'+str(shear)+'_'+str(photoz)+'_pofz.fits', clobber=True)

	for i in range(Nbin_z_s):
		zbin_mask = ((z_pofz_bin+0.05)>Z1_s[i])*((z_pofz_bin+0.05)<=Z2_s[i])
		n_weight = np.zeros(len(ids))
		for j in range(len(z_pofz_bin[zbin_mask])):
			n_weight += pofz_s[mask][ids][:, int(z_pofz_bin[zbin_mask][j]/0.1)]

		c1 = pf.Column(name='RA', format='E', array=RA[mask][ids])
		c2 = pf.Column(name='DEC', format='E', array=DEC[mask][ids])
		c3 = pf.Column(name='S1', format='E', array=G1[mask][ids]*(-1))
		c4 = pf.Column(name='S2', format='E', array=G2[mask][ids]*(-1))
		c5 = pf.Column(name='E1', format='E', array=G1[mask][ids]*(-1)+e1_ran)
		c6 = pf.Column(name='E2', format='E', array=G2[mask][ids]*(-1)+e2_ran)
		c7 = pf.Column(name='Z_true', format='E', array=Z[mask][ids])
		c8 = pf.Column(name='Z_mean', format='E', array=zmean_s[mask][ids])
		c9 = pf.Column(name='NZ_weight', format='E', array=n_weight)
		c10 = pf.Column(name='KAPPA', format='E', array=K[mask][ids])

		CC = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
		hdu = pf.new_table(CC, nrows=len(Z[mask][ids]))
		hdu.writeto(temp_dir_sim+'background_mice_sv_'+str(shear)+'_'+str(photoz)+'_'+str(i)+'_pofz.fits', clobber=True)

mask = (MAG>maglim1)*(MAG<maglim2)*(RA<ramax)*(RA>ramin)*(DEC<decmax)*(DEC>decmin)
nrows = len(RA[mask])

c1 = pf.Column(name='RA', format='E', array=RA[mask])
c2 = pf.Column(name='DEC', format='E', array=DEC[mask])
c3 = pf.Column(name='Z_true', format='E', array=Z[mask]) 
c4 = pf.Column(name='Z_mean', format='E', array=zmean_l[mask]) 

CC = [c1, c2, c3, c4]
hdu = pf.new_table(CC, nrows=nrows)
hdu.writeto(temp_dir_sim+'foreground_mice_sv.fits', clobber=True)


