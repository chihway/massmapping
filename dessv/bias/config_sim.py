
import numpy as np

# common parameters

pix = 5.0
maglim1 = 18.0
maglim2 = 22.5
smooth = 50.0
N_ran = 1 #10?
ran_seed = 1234

Z1_l = np.array([0.0,0.2,0.4,0.6,0.8,1.0])
Z2_l = np.array([0.2,0.4,0.6,0.8,1.0,1.2])
dz_bin = Z2_l[0]-Z1_l[0]
z_l = (Z1_l+Z2_l)/2
Nbin_z_l = len(z_l)
b_2pcf = 1.38-0.95*z_l+1.36*z_l**2
Z1_s = np.array([0.2,0.4,0.6,0.8,1.0,1.2])
Z2_s = np.array([0.4,0.6,0.8,1.0,1.2,1.4])
z_s = (Z1_s+Z2_s)/2
Nbin_z_s = len(z_s)
z_pofz_bin = np.arange(18)*0.1

cosmo = {'omega_M_0':0.25, 'omega_lambda_0':0.75, 'omega_k_0':0.0, 'h':0.7, 'sigma_8' : 0.8, 'omega_b_0' : 0.044, 'n' : 0.95}

smooth_kernal = 'box'
jk_area = 10. # deg^2
njk = 20
# jk_factor1 = 1./0.834942454678
# jk_factor2 = 1./0.611366938348
jk_factor1 = 1./0.91
jk_factor2 = 1./0.93
bias_type = 'bias_denoise' 
smooth_scale = int(smooth/pix)
mask_edge_arcmin = smooth*1.0

temp_dir_data = '/Users/chihwaychang/Desktop/Work/bias/kappa_bias/temp_files/data/'
temp_dir_sim = '/Users/chihwaychang/Desktop/Work/bias/kappa_bias/temp_files/sim/'
fig_dir='/Users/chihwaychang/Desktop/Work/bias/kappa_bias/figs/'

# sim specific

ramax = 30.0
ramin = 0.0
ra_ref = ramin + (ramax-ramin)/2
decmin = 0.0
decmax = 30.0
Nbin_ra = int((ramax-ramin)*60./pix)
Nbin_dec = int((decmax-decmin)*60./pix)

seed = 100
mask_type = 0

# whick type of redshifts are we using?
zmean_type = 0
ztrue_type = 1
nofz_type = 0

# middle
# mask_c1 = 80
# mask_c2 = 290
# mask_c3 = 80
# mask_c4 = 255

# 1
# mask_flag = 1
# mask_c1 = 30
# mask_c2 = 240
# mask_c3 = 30
# mask_c4 = 205

# 2
# mask_flag = 2
# mask_c1 = 120
# mask_c2 = 330
# mask_c3 = 30
# mask_c4 = 205

# 3
# mask_flag = 3
# mask_c1 = 30
# mask_c2 = 240
# mask_c3 = 155
# mask_c4 = 330

#4
mask_flag = 4
mask_c1 = 120
mask_c2 = 330
mask_c3 = 155
mask_c4 = 330

photoz = 'skynet'
shear = 'ngmix'

cat_gold_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv18/des_sv_wl_info.fits'
# cat_gold_zmean_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv16/flat_skynet_z.fits'
cat_gold_zmean_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv18/flat_skynet_z.fits'
cat_gold_pofz_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv18/flat_skynet_pofz.fits'
cat_gold_pofz_binned_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv18/flat_skynet_pofz_binned.fits'
cat_wl_info_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv18/des_sv_wl_info.fits'
cat_wl_im3shape_name ='/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv18/des_sv_wl_im3shape.fits' 
cat_wl_ngmix_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv18/des_sv_wl_ngmix.fits'
depth_map_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/sva1_gold_1.0.2-4_nside4096_nest_i_auto_weights.fits'

cat_mice_coord_name = '/Volumes/astro/refreg/temp/cchang/MICE_for_des_bias/MICE_coord.fits'  #901
cat_mice_shear_name = '/Volumes/astro/refreg/temp/cchang/MICE_for_des_bias/MICE_shear.fits'  #901
#cat_bccufig_pofz_source_name = '/Users/chihwaychang/Google Drive/Work/catalogs/MICE/bccufig_pofz_source.fits'
# cat_bccufig_pofz_binned_source_name = '/Users/chihwaychang/Google Drive/Work/catalogs/MICE/bccufig_pofz_source.fits'
# cat_bccufig_pofz_binned_lens_name = '/Users/chihwaychang/Google Drive/Work/catalogs/MICE/bccufig_pofz_lens.fits'
cat_mice_pz_source_name = '/Volumes/astro/refreg/temp/cchang/MICE_for_des_bias/MICE_zp_source.fits'
cat_mice_pz_lens_name = '/Volumes/astro/refreg/temp/cchang/MICE_for_des_bias/MICE_zp_lens.fits'

ran_name = temp_dir_sim+'random_gal_sim.fits'

