"""
Make convergence maps and other diagnostic maps from BCC simulations.
"""

import numpy as np
import astropy.io.fits as pf
import fitsio
import healpy as hp
import sys
sys.path.append('../utils')
from massmapping_utils import *

buzzardid = int(sys.argv[1])
zbin = int(sys.argv[2])

if buzzardid>=1 and buzzardid<=6:
    name2 = 'Buzzard1_v1.3'
if buzzardid>=7 and buzzardid<=12:
    name2 = 'Buzzard2_v1.3'
if buzzardid>=13 and buzzardid<=18:
    name2 = 'Buzzard3_v1.3'

name1 = 'buzzard_v1.3'
name3 = 'buzzard'
Buzzard_ids = ['x', 'a', 'b', 'c', 'd', 'e', 'f', 'a', 'b', 'c', 'd', 'e', 'f', 'a', 'b', 'c', 'd', 'e', 'f']

nside = 1024 # 3 arcmin pixels
lmax = 2*nside

source = fitsio.FITS('sims/'+str(name1)+'/mock'+str(buzzardid)+'/'+str(name2)+'_'+str(Buzzard_ids[buzzardid])+'_gold.fits')[1]
sample_mask = np.where(source['wl-sample'][:]==1)
ra = source['ra'][sample_mask]
dec = source['dec'][sample_mask]
source = 0

source_z = fitsio.FITS('sims/'+str(name1)+'/mock'+str(buzzardid)+'/'+str(name2)+'_'+str(Buzzard_ids[buzzardid])+'_pz.fits')[1]
zp = source_z['mean-z'][sample_mask]
zmc = source_z['mc-z'][sample_mask]
z = source_z['redshift'][sample_mask]
source_z = 0

source_shape = fitsio.FITS('sims/'+str(name1)+'/mock'+str(buzzardid)+'/'+str(name2)+'_'+str(Buzzard_ids[buzzardid])+'_shape.fits')[1]
e1 = source_shape['e1'][sample_mask]*(-1)
e2 = source_shape['e2'][sample_mask]*(-1)
g1 = source_shape['g1'][sample_mask]*(-1)
g2 = source_shape['g2'][sample_mask]*(-1)
k = source_shape['kappa'][sample_mask]
source_shape = 0

if zbin==0:
    zmin = 0.2
    zmax = 0.43
    print 'zbin 0'
if zbin==1:
    zmin = 0.43
    zmax = 0.63
    print 'zbin 1'
if zbin==2:
    zmin = 0.63
    zmax = 0.9
    print 'zbin 2'
if zbin==3:
    zmin = 0.9
    zmax = 1.3
    print 'zbin 3'
if zbin==4:
    zmin = 0.2
    zmax = 0.63
    print 'zbin 4'
if zbin==5:
    zmin = 0.63
    zmax = 1.3
    print 'zbin 5'
if zbin==6:
    zmin = 0.2
    zmax = 1.3
    print 'zbin 6'

mask = (zp>=zmin)*(zp<zmax)*((e1**2+e2**2)<100)
ra = ra[mask]
dec = dec[mask]
z = z[mask]
zp = zp[mask]
zmc = zmc[mask]
e1 = e1[mask]
e2 = e2[mask]
g1 = g1[mask]
g2 = g2[mask]
ids_ran = np.arange(len(z))
np.random.shuffle(ids_ran)
ra_ran = ra[ids_ran]
dec_ran = dec[ids_ran]
k = k[mask]
sens = ra*0.0 + 1.0
w = ra*0.0 + 1.0
print 'Ngal=', len(ra)

e1_data = fitsio.FITS('mcal_e1.fits')[1]['e1'][:]*(-1)
e2_data = fitsio.FITS('mcal_e2.fits')[1]['e2'][:]

ids_ran = np.arange(len(e1_data))
np.random.shuffle(ids_ran)

if zbin==6:
    N_buzzard = len(ra)*1.0
    N_data = 21404317.0
    SN_factor = (N_data/N_buzzard)**0.5 * (np.var(e1_data)+np.var(e2_data))**0.5 / (np.var(e1_data)+np.var(e2_data)-np.var(g1)-np.var(g2))**0.5 
else: 
    SN_factor = 1.1* (np.var(e1_data)+np.var(e2_data))**0.5 / (np.var(e1_data)+np.var(e2_data)-np.var(g1)-np.var(g2))**0.5

print SN_factor

e1 = g1 + e1_data[ids_ran[:len(g1)]]/SN_factor #np.random.normal(0.0, 0.25, len(g1))
e2 = g2 + e2_data[ids_ran[:len(g2)]]/SN_factor #np.random.normal(0.0, 0.25, len(g2))
# correct for lower number density

map_n, map_nw, map_e1 = make_healpix_map(ra, dec, e1, w, nside)
map_n, map_nw, map_e2 = make_healpix_map(ra, dec, e2, w, nside)
map_e1[map_n!=0] = map_e1[map_n!=0]/map_nw[map_n!=0]
map_e2[map_n!=0] = map_e2[map_n!=0]/map_nw[map_n!=0]

map_n, map_nw, map_g1 = make_healpix_map(ra, dec, g1, w, nside)
map_n, map_nw, map_g2 = make_healpix_map(ra, dec, g2, w, nside)
map_g1[map_n!=0] = map_g1[map_n!=0]/map_nw[map_n!=0]
map_g2[map_n!=0] = map_g2[map_n!=0]/map_nw[map_n!=0]

map_n, map_nw, map_e1_ran = make_healpix_map(ra_ran, dec_ran, e1, w, nside)
map_n, map_nw, map_e2_ran = make_healpix_map(ra_ran, dec_ran, e2, w, nside)
map_e1_ran[map_n!=0] = map_e1_ran[map_n!=0]/map_nw[map_n!=0]
map_e2_ran[map_n!=0] = map_e2_ran[map_n!=0]/map_nw[map_n!=0]

map_n, map_nw, map_g1_ran = make_healpix_map(ra_ran, dec_ran, g1, w, nside)
map_n, map_nw, map_g2_ran = make_healpix_map(ra_ran, dec_ran, g2, w, nside)
map_g1_ran[map_n!=0] = map_g1_ran[map_n!=0]/map_nw[map_n!=0]
map_g2_ran[map_n!=0] = map_g2_ran[map_n!=0]/map_nw[map_n!=0]

map_n, map_nw, map_k = make_healpix_map(ra, dec, k, w, nside)
map_k[map_n!=0] = map_k[map_n!=0]/map_nw[map_n!=0]
    
survey_mask = map_n.copy()
survey_mask[survey_mask>0] = 1

e2kappa_mask, e2kappa_map_alm, e2E_map, e2B_map = g2k_sphere(map_e1*0.0, map_e1, map_e2, survey_mask, nside=nside, lmax=lmax)
g2kappa_mask, g2kappa_map_alm, g2E_map, g2B_map = g2k_sphere(map_g1*0.0, map_g1, map_g2, survey_mask, nside=nside, lmax=lmax)
e2kappa_ran_mask, e2kappa_ran_map_alm, e2E_ran_map, e2B_ran_map = g2k_sphere(map_e1_ran*0.0, map_e1_ran, map_e2_ran, survey_mask, nside=nside, lmax=lmax)
g2kappa_ran_mask, g2kappa_ran_map_alm, g2E_ran_map, g2B_ran_map = g2k_sphere(map_g1_ran*0.0, map_g1_ran, map_g2_ran, survey_mask, nside=nside, lmax=lmax)
kappa_mask, kappa_map_alm, kappaE_map, kappaB_map = g2k_sphere(map_k, map_g1*0.0, map_g2*0.0, survey_mask, nside=nside, lmax=lmax)
    
nofz_true = np.histogram(z, range=(0.0,3.0), bins=300)
nofz_mc = np.histogram(zmc, range=(0.0,3.0), bins=300)
area = np.sum(survey_mask)*1.0/len(survey_mask)*(180.0/np.pi)**2*4*np.pi

# save info file
np.savez(+str(name3)+'_y1_'+str(buzzardid)+'_'+str(zbin)+'_info.npz',
ngal=len(e1), e1_std=e1.std(), e2_std=e2.std(), mean_z_true=z.mean(), mean_z=zp.mean(),
nofz_bin=(nofz_true[1][1:]+nofz_true[1][:-1])/2, nofz=nofz_true[0], nofz_mc=nofz_mc[0], area=area)

# write all the maps
names = ['k', 'kalm', 'Nsource', 'g2kE', 'g2kB', 'e2kE', 'e2kB', 'g2kE_ran', 'g2kB_ran', 'e2kE_ran', 'e2kB_ran', 'mask']
Maps = [map_k, kappa_map_alm, map_n, g2E_map, g2B_map, e2E_map, e2B_map, g2E_ran_map, g2B_ran_map, e2E_ran_map, e2B_ran_map, survey_mask]

# write all the maps
for i in range(len(names)):
    fits = fitsio.FITS(str(name3)+'_y1_'+str(buzzardid)+'_'+str(zbin)+'_'+str(names[i])+'.fits','rw',clobber=True)
    output = np.zeros(hp.nside2npix(nside), dtype=[(names[i],'f8')])
    output[names[i]] = Maps[i]
    fits.write(output)


