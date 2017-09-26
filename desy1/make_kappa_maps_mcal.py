
"""
Make MetaCalibration convergence maps and other diagnostic maps.
"""

import numpy as np
import fitsio
import numpy as np
import sys
sys.path.append('../utils')
from massmapping_utils import *

nside = 1024  # 3 arcmin pixels
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

# read files
mcal_id = fitsio.FITS('y1a1_metacal_id.fits')[1]
mcal_ellip = fitsio.FITS('y1a1_metacal_ellip.fits')[1]
mcal_psf = fitsio.FITS('y1a1_metacal_psf.fits')[1]
mcal_z = fitsio.FITS('y1a1_metacal_z.fits')[1]

flags = mcal_id['flags_select'][:]
flags_1p = mcal_id['flags_select_1p'][:]
flags_1m = mcal_id['flags_select_1m'][:]
flags_2p = mcal_id['flags_select_2p'][:]
flags_2m = mcal_id['flags_select_2m'][:]

ra = mcal_id['ra'][:]
dec = mcal_id['dec'][:]
z_0 = mcal_z['mean_z'][:]
z_1p = mcal_z['mean_z_1p'][:]
z_1m = mcal_z['mean_z_1m'][:]
z_2p = mcal_z['mean_z_2p'][:]
z_2m = mcal_z['mean_z_2m'][:]
w = np.ones(len(ra))

# if random, scramble the ra/dec's 
if random==1:
    ids_ran = np.arange(len(ra))
    np.random.shuffle(ids_ran)
    ra = ra[ids_ran]
    dec = dec[ids_ran]

mask0 = np.where((flags==0)*(z_0>=zmin)*(z_0<zmax)*(ra>ra_min)*(ra<ra_max))
mask_w1p = np.where((flags_1p==0)*(z_1p>=zmin)*(z_1p<zmax)*(ra>ra_min)*(ra<ra_max))
mask_w1m = np.where((flags_1m==0)*(z_1m>=zmin)*(z_1m<zmax)*(ra>ra_min)*(ra<ra_max))
mask_w2p = np.where((flags_2p==0)*(z_2p>=zmin)*(z_2p<zmax)*(ra>ra_min)*(ra<ra_max))
mask_w2m = np.where((flags_2m==0)*(z_2m>=zmin)*(z_2m<zmax)*(ra>ra_min)*(ra<ra_max))
flags = 0
flags_1p = 0
flags_1m = 0
flags_2p = 0
flags_2m = 0
z_1p = 0
z_1m = 0
z_2p = 0
z_2m = 0

# calcualte metacalibration factors
dgamma = 2*0.01

R11  = (np.mean(mcal_ellip['e1_1p'][mask0]) - np.mean(mcal_ellip['e1_1m'][mask0]))/dgamma
R11s = (np.mean(mcal_ellip['e1'][mask_w1p]) - np.mean(mcal_ellip['e1'][mask_w1m]))/dgamma
R11tot = R11 + R11s
print R11, R11s
print 'R11tot mean', R11tot

R22  = (np.mean(mcal_ellip['e2_2p'][mask0]) - np.mean(mcal_ellip['e2_2m'][mask0]))/dgamma
R22s = (np.mean(mcal_ellip['e2'][mask_w2p]) - np.mean(mcal_ellip['e2'][mask_w2m]))/dgamma
R22tot = R22 + R22s
print R22, R22s
print 'R22tot mean', R22tot

ra = ra[mask0]
dec = dec[mask0]
w = w[mask0]
e1 = mcal_ellip['e1'][mask0]
e2 = mcal_ellip['e2'][mask0]
zmc_0 = mcal_z['z_mc'][mask0]
psf_e1 = mcal_psf['psf_e1'][mask0]
psf_e2 = mcal_psf['psf_e2'][mask0]
psf_size = mcal_psf['psf_size'][mask0]

map_n, map_nw, map_E1 = make_healpix_map(ra, dec, e1, w, nside=nside)
map_n, map_nw, map_E2 = make_healpix_map(ra, dec, e2, w, nside=nside)
map_E1[map_nw!=0] /= map_nw[map_nw!=0]
map_E2[map_nw!=0] /= map_nw[map_nw!=0]
map_E1 /= R11tot
map_E2 /= R22tot

# calibrate, mean-subtract, and adjust sign of ellipticities
mean_e1 = np.mean(e1)/R11tot
mean_e2 = np.mean(e2)/R22tot
map_E1[map_nw!=0] -= mean_e1
map_E2[map_nw!=0] -= mean_e2
map_E1 = map_E1*(-1)

# calculate area and n(z)
survey_mask = map_n.copy()
survey_mask[survey_mask>0] = 1
nofz = np.histogram(zmc_0, range=(0.0,3.0), bins=300)
area = np.sum(survey_mask)*1.0/len(survey_mask)*(180.0/np.pi)**2*4*np.pi

# spin transformation. NOTE: E1 is flipped!
kappa_mask, kappa_map_alm, E_map, B_map = g2k_sphere(0.0*map_E1, map_E1, map_E2, survey_mask, nside=nside, lmax=lmax)

# write maps, amd meta info
if random==1:
    names = ['kE_ran', 'kB_ran']
    Maps = [E_map, B_map]
    for i in range(len(names)):
        fits = fitsio.FITS('y1a1_spt_mcal_'+str(zbin)+'_'+str(names[i])+'.fits','rw',clobber=True)
        output = np.zeros(hp.nside2npix(nside), dtype=[(names[i],'f8')])
        output[names[i]] = Maps[i]
        fits.write(output)
else:
    np.savez('y1a1_spt_mcal_'+str(zbin)+'_info.npz',
    ngal=len(e1), e1_std=e1.std(), e2_std=e2.std(), mean_R11=R11tot, mean_R22=R22tot, mean_z=z_0[mask0].mean(),
    nofz_bin=(nofz[1][1:]+nofz[1][:-1])/2, nofz_mc=nofz[0], area=area)

    # also make PSF E1, E2, SIZE, kE, kB and shear SNR maps
    map_n, map_nw, map_PSF_E1 = make_healpix_map(ra, dec, psf_e1, w, nside=nside)
    map_n, map_nw, map_PSF_E2 = make_healpix_map(ra, dec, psf_e2, w, nside=nside)
    map_n, map_nw, map_PSF_SIZE = make_healpix_map(ra, dec, psf_size, w, nside=nside)

    map_PSF_E1[map_nw!=0] /= map_nw[map_nw!=0]
    map_PSF_E2[map_nw!=0] /= map_nw[map_nw!=0]
    map_PSF_SIZE[map_nw!=0] /= map_nw[map_nw!=0]

    # spin transformation. NOTE: E1 is flipped!
    kappa_mask, kappa_map_alm, PSF_E_map, PSF_B_map = g2k_sphere(0.0*map_E1, map_PSF_E1, map_PSF_E2, survey_mask, nside=nside, lmax=lmax)
    map_PSF_E1 = map_PSF_E1*(-1)

    names = ['kE', 'kB', 'Nsource', 'mask', 'E1', 'E2', 'PSF_kE', 'PSF_kB', 'PSF_E1', 'PSF_E2', 'PSF_SIZE']
    Maps = [E_map, B_map, map_n, survey_mask, map_E1, map_E2, PSF_E_map, PSF_B_map, map_PSF_E1, map_PSF_E2, map_PSF_SIZE]

    for i in range(len(names)):
        fits = fitsio.FITS('y1a1_spt_mcal_'+str(zbin)+'_'+str(names[i])+'.fits','rw',clobber=True)
        output = np.zeros(hp.nside2npix(nside), dtype=[(names[i],'f8')])
        output[names[i]] = Maps[i]
        fits.write(output)



