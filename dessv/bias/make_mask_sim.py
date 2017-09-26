
"""
This script makes the masks that will be used for the bias 
measurement for sims, these masks are:

1) mask_mice_map.npz: mask multiplied to all data maps, 
gamma/kappa/gamma_g/kappa_g, we do not use data outside 
these masked areas. This map is the AND of the ngal>0 map 
of foreground and background ngal>0 masks.

2) mask_mice_bias.npz: take mask_map.npz and grow the 
mask so that the edges of the original mask are masked. 
The additional masked region is determined by the smoothing 
scale used in the main analysis.

3) mask_mice_sv_map.npz: adding SV mask on top.

4) mask_mice_sv_bias.npz: adding SV mask on top.
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np 
import pyfits as pf 
import sys, os
import pylab as mplot
import healpy as hp
sys.path.append('../utils')
from utils import smooth_map_boxcar, invert_mask, grow_mask
import config_sim

for name in [name for name in dir(config_sim) if not name.startswith("__")]:
    globals()[name] = getattr(config_sim, name)

# lens mask ########

print('make lens mask...')

D = pf.open(temp_dir_sim+'foreground_mice_sv.fits')[1].data
ra = D['RA']
dec = D['DEC']

ra_p = ra_ref + (ra-ra_ref)*np.cos(dec/180.*np.pi)
N2d, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)))
mask_mice_lens = N2d*1.0
mask_mice_lens[mask_mice_lens>0]=1

# source mask ########

print('make source mask...')

D = pf.open(temp_dir_sim+'background_mice_sv_'+str(shear)+'_'+str(photoz)+'_0.fits')[1].data 
ra = D['RA']
dec = D['DEC']

ra_p = ra_ref + (ra-ra_ref)*np.cos(dec/180.*np.pi)
N2d, edges = np.histogramdd(np.array([dec, ra_p]).T, bins=(Nbin_dec, Nbin_ra), range=((decmin, decmax),(ramin, ramax)))

mask_mice_source = N2d*1.0
mask_mice_source[mask_mice_source>0]=1

print('combine all masks...')

# this is the mask we want to use for all the map construction
mask_mice_total = mask_mice_lens*mask_mice_source
np.savez(temp_dir_sim+'mask_mice_map.npz', mask=mask_mice_total)

# this is the mask we want to use for all bias calculation
mask_mice_bias = grow_mask(mask_mice_total, Nbin_ra, Nbin_dec, fill=3, edge=mask_edge_arcmin, pix=pix)
mask_mice_bias[mask_mice_total==0] = 0
mask_mice_bias[mask_mice_bias<1] = 0
np.savez(temp_dir_sim+'mask_mice_bias.npz', mask=mask_mice_bias)

print('do the same adding SV mask...')
# now add SV mask on top
map_map_sv = np.load(temp_dir_data+'mask_map.npz')['mask']
mask_mice_sv_total = mask_mice_total*0.0
mask_mice_sv_total[mask_c1:mask_c2,mask_c3:mask_c4] = map_map_sv*1.0
np.savez(temp_dir_sim+'mask_mice_sv_map_'+str(mask_flag)+'.npz', mask=mask_mice_sv_total)

map_bias_sv = np.load(temp_dir_data+'mask_bias.npz')['mask']
mask_mice_sv_bias = mask_mice_total*0.0
mask_mice_sv_bias[mask_c1:mask_c2,mask_c3:mask_c4] = map_bias_sv*1.0
np.savez(temp_dir_sim+'mask_mice_sv_bias_'+str(mask_flag)+'.npz', mask=mask_mice_sv_bias)

mplot.figure(figsize=(16,12))
mplot.subplot(231)
mplot.imshow(mask_mice_lens)
mplot.title('MICE lens mask (ngal>0)')
mplot.subplot(232)
mplot.imshow(mask_mice_source)
mplot.title('MICE source mask (ngal>0)')
mplot.subplot(233)
mplot.imshow(mask_mice_total)
mplot.title('MICE total mask')
mplot.subplot(234)
mplot.imshow(mask_mice_bias)
mplot.title('MICE bias mask')
mplot.subplot(235)
mplot.imshow(mask_mice_sv_total)
mplot.title('MICE total SV mask')
mplot.subplot(236)
mplot.imshow(mask_mice_sv_bias)
mplot.title('MICE bias SV mask')
mplot.tight_layout()

mplot.show()




