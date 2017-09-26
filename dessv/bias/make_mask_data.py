
"""
This script makes the masks that will be used for the bias 
measurement for data, these masks are:

1) mask_map.npz: mask multiplied to all data maps, 
gamma/kappa/gamma_g/kappa_g, we do not use data outside 
these masked areas. This map is the AND of the ngal>0 map 
of Gold and the shear catalog and a i>22.5 depth map. 
- all maps are repixelated into 5 arcmin pixels
- the i<22 depth mask is repixelated from a nside=4096 
healpix map, and the new pixels are set to 1 if there are 
>50 percent of the healpix pixels ==1. 

2) mask_bias.npz: take mask_map.npz and grow the mask so 
that the edges of the original mask are masked. The 
additional masked region is determined by the smoothing 
scale used in the main analysis.

"""

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np 
import pyfits as pf 
import sys, os
import pylab as mplot
import healpy as hp
sys.path.append('../utils')
from utils import smooth_map_boxcar, invert_mask, grow_mask
import config_data

for name in [name for name in dir(config_data) if not name.startswith("__")]:
    globals()[name] = getattr(config_data, name)

# lens mask ########

print('make lens mask...')

cat_gold = pf.open(temp_dir_data+'benchmark_masked.fits')[1].data
ra = cat_gold['RA']
dec = cat_gold['DEC']
ra_p = ra_ref + (ra-ra_ref)*np.cos(dec/180.*np.pi)

N2d, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)))
mask_lens = N2d*1.0
mask_lens[mask_lens>0]=1

# source mask ########

print('make source mask...')

cat_shear = pf.open(temp_dir_data+'background_'+str(shear)+'_tomo_zmean_'+str(photoz)+'_0.fits')[1].data 
# since we're using full pofz, the ngal>0 mask is the same for all bins

ra = cat_shear['RA']
dec = cat_shear['DEC']
# nz_weight = cat_shear['NZ_weight']

ra_p = ra_ref + (ra-ra_ref)*np.cos(dec/180.*np.pi)
N2d, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)))

mask_source = N2d*1.0
mask_source[mask_source>0]=1

# depth mask ########

print('make depth mask...')

depth = hp.read_map(depth_map_name)

pix_h = range(hp.nside2npix(4096))
theta, phi = hp.pix2ang(4096, pix_h)
npixel = pix**2/(hp.nside2resol(4096)/np.pi*180*60)**2

dec = 90.0 - theta*180/np.pi
ra = phi*180./np.pi
ra[ra>180] = ra[ra>180]-360.0

mask = (ra>60)*(ra<98)*(dec<-38)*(depth>=maglim2)*(depth!=(-1.63750000e+30))
ra = np.array(ra[mask])
dec = np.array(dec[mask])
depth = np.array(depth[mask])

ra_p = ra_ref + (ra-ra_ref)*np.cos(dec/180.*np.pi)
N2d, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)))

mask_depth = N2d*1.0
# mplot.figure()
# mplot.hist(mask_depth.flatten()*1.0/npixel, range=(0,1.2), bins=100, histtype='step')
# mplot.show()

print(np.sum(mask_depth[mask_depth>=(0.5*npixel)])*1.0/npixel)
print(len(mask_depth[mask_depth>=(0.5*npixel)]))
temp_mask_depth = mask_depth*1.0
mask_depth[mask_depth<(0.5*npixel)]=0
mask_depth[mask_depth>=(0.5*npixel)]=1

print('combine all masks...')

# this is the mask we want to use for all the map construction
mask_total = mask_lens*mask_source*mask_depth
mask_total[11] = 0 
# this pixel is at the edge of the mask and induces strange problems since it's not in the 
# depth mask directly, so we manually mask it out. All other edges are taken care of...

np.savez(temp_dir_data+'mask_map.npz', mask=mask_total)
np.savez(temp_dir_data+'mask_fraction.npz', fraction=(temp_mask_depth*1.0/npixel)*mask_total)

# this is the mask we want to use for all bias calculation
mask_bias = grow_mask(mask_total, Nbin_ra, Nbin_dec, fill=3, edge=mask_edge_arcmin, pix=pix)
mask_bias[mask_total==0] = 0
mask_bias[mask_bias<1] = 0
np.savez(temp_dir_data+'mask_bias.npz', mask=mask_bias)

mplot.figure(figsize=(16,12))
mplot.subplot(231)
mplot.imshow(mask_lens)
mplot.title('lens mask (ngal>0)')
mplot.subplot(232)
mplot.imshow(mask_source)
mplot.title('source mask (ngal>0)')
mplot.subplot(233)
mplot.imshow(mask_depth)
mplot.title('depth mask (i>22.5)')
mplot.subplot(234)
mplot.imshow(mask_total)
mplot.title('total mask for maps')
mplot.subplot(235)
mplot.imshow(mask_bias)
mplot.title('total mask for bias')
mplot.tight_layout()

mplot.show()


