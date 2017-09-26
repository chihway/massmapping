"""
Make Im3shape convergence maps and other diagnostic maps.
"""

import numpy as np
import healpy as hp
import sys
sys.path.append('../utils')
from massmapping_utils import *
import fitsio

nside = 1024 #1024 # 3 arcmin pixels
lmax = 2*nside

zbin = int(sys.argv[1]) # z bins
random = int(sys.argv[2]) # generate random map or not

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

source_id = fitsio.FITS('y1a1_im3shape_id.fits')[1]
source_z = fitsio.FITS('y1a1_im3shape_z.fits')[1]
source_ellip = fitsio.FITS('y1a1_im3shape_ellip.fits')[1]
source_psf = fitsio.FITS('y1a1_im3shape_psf.fits')[1]

mask = (source_z['mean_z'][:]>=zmin)*(source_z['mean_z'][:]<zmax)
    
mean_m_0 = np.mean(source_ellip['m'][:][mask])
mean_sens = np.sum((1+source_ellip['m'][:][mask])*source_ellip['weight'][:][mask])/np.sum(source_ellip['weight'][:][mask])

ra = source_id['ra'][:][mask]
dec = source_id['dec'][:][mask]
e1 = source_ellip['e1'][:][mask] 
e2 = source_ellip['e2'][:][mask] 
e1 = (e1 - source_ellip['c1'][:][mask])*(-1)
e2 = (e2 - source_ellip['c2'][:][mask])
sens = (1.0+mean_m_0)*np.ones(len(ra))
w = source_ellip['weight'][:][mask]
z = source_z['mean_z'][:][mask]
zmc = source_z['z_mc'][:][mask]
psf_e1 = source_psf['psf_e1'][:][mask]*(-1)
psf_e2 = source_psf['psf_e2'][:][mask]
psf_size = source_psf['psf_size'][:][mask]

mean_e1 = np.sum(e1*source_ellip['weight'][:][mask])/np.sum(source_ellip['weight'][:][mask])/mean_sens
mean_e2 = np.sum(e2*source_ellip['weight'][:][mask])/np.sum(source_ellip['weight'][:][mask])/mean_sens

# if random, scramble ra/dec
if random==1:
    ids_ran = np.arange(len(e1))
    np.random.shuffle(ids_ran)
    ra = ra[ids_ran]
    dec = dec[ids_ran]

# make main maps
map_n, map_nw, map_e1 = make_healpix_map(ra, dec, e1, w)
map_n, map_nw, map_e2 = make_healpix_map(ra, dec, e2, w)
map_n, map_ne, map_sens = make_healpix_map(ra, dec, sens, w)
map_e1[map_nw!=0] /= map_sens[map_nw!=0]
map_e2[map_nw!=0] /= map_sens[map_nw!=0]
map_e1[map_nw!=0] -= mean_e1
map_e2[map_nw!=0] -= mean_e2

survey_mask = map_n.copy()
survey_mask[survey_mask>0] = 1

kappa_mask, kappa_map_alm, E_map, B_map = g2k_sphere(map_e1*0.0, map_e1, map_e2, survey_mask, nside=nside, lmax=lmax)

# write all the maps
if random==1:
    names = ['kE_ran', 'kB_ran']
    Maps = [E_map, B_map]
    for i in range(len(names)):
        fits = fitsio.FITS('y1a1_spt_im3shape_'+str(zbin)+'_'+str(names[i])+'.fits','rw',clobber=True)
        output = np.zeros(hp.nside2npix(nside), dtype=[(names[i],'f8')])
        output[names[i]] = Maps[i]
        fits.write(output)

else:
    area = np.sum(survey_mask)*1.0/len(survey_mask)*(180.0/np.pi)**2*4*np.pi
    nofz = np.histogram(zmc, range=(0.0,3.0), bins=300)

    a1 = np.sum(w**2 * e1**2)
    a2 = np.sum(w**2 * e2**2)
    b  = np.sum(w**2)
    c1 = np.sum(w * sens)
    c2 = np.sum(w*sens)
    d  = np.sum(w)
    sigma_e = np.sqrt( (a1/c1**2 + a2/c2**2) * (d**2/b) ) 

    np.savez('y1a1_spt_im3shape_'+str(zbin)+'_info.npz',
    ngal=len(e1), e1_std=np.std(e1), e2_std=np.std(e2), mean_sens=np.mean(sens), mean_z=z.mean(), sigmae=sigma_e,
    nofz_bin=(nofz[1][1:]+nofz[1][:-1])/2, nofz=nofz[0], area=area)

    # also make PSF E1, E2, SIZE, kE, kB and shear SNR maps
    map_nn, map_nw, map_psf_e1 = make_healpix_map(ra, dec, psf_e1, w)
    map_nn, map_nw, map_psf_e2 = make_healpix_map(ra, dec, psf_e2, w)
    map_nn, map_ne, map_psf_size = make_healpix_map(ra, dec, psf_size, w)

    map_psf_e1[map_nw!=0] /= map_nw[map_nw!=0]
    map_psf_e2[map_nw!=0] /= map_nw[map_nw!=0]
    map_psf_size[map_nw!=0] /= map_nw[map_nw!=0]

    kappa_mask, kappa_map_alm, PSF_E_map, PSF_B_map = spin_transform_nosm(map_e1*0.0, map_psf_e1, map_psf_e2, survey_mask, nside=nside, lmax=lmax)

    names = ['kE', 'kB', 'Nsource', 'mask', 'E1', 'E2', 'PSF_kE', 'PSF_kB', 'PSF_E1', 'PSF_E2', 'PSF_SIZE']
    Maps = [E_map, B_map, map_n, survey_mask, map_e1, map_e2, PSF_E_map, PSF_B_map, map_psf_e1, map_psf_e2, map_psf_size]
    
    for i in range(len(names)):
        fits = fitsio.FITS('y1a1_spt_im3shape_'+str(zbin)+'_'+str(names[i])+'.fits','rw',clobber=True)
        output = np.zeros(hp.nside2npix(nside), dtype=[(names[i],'f8')])
        output[names[i]] = Maps[i]
        fits.write(output)

