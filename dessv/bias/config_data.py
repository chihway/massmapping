
import numpy as np

# common parameters

pix = 5.0
maglim1 = 18.0
maglim2 = 22.5
smooth = 50.0 #100?
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
jk_factor1 = 1./0.91
jk_factor2 = 1./0.93
# jk_factor1 = 1./0.834942454678
# jk_factor2 = 1./0.611366938348
smooth_scale = int(smooth/pix)
mask_edge_arcmin = smooth*1.0
bias_type = 'bias_denoise' 

temp_dir_data = '/Users/chihwaychang/Desktop/Work/bias/kappa_bias/temp_files/data/'
temp_dir_sim = '/Users/chihwaychang/Desktop/Work/bias/kappa_bias/temp_files/sim/'
fig_dir='/Users/chihwaychang/Desktop/Work/bias/kappa_bias/figs/'

# data specific

#cal_factor = np.array([ 1.29793441,1.20372353,1.15868317,1.21999684,1.17800161,1.17800161])
cal_factor = np.array([ 1.0,1.0,1.0,1.0,1.0,1.0])

ra_ref = 71.0
ramin = 65.6513137817
ramax = 80.2797393799
decmin = -60.9955749511
decmax = -43.4159164428
Nbin_ra = int((ramax-ramin)*60./pix)
Nbin_dec = int((decmax-decmin)*60./pix)

photoz = 'skynet'
cat_gold_zmean_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv16/flat_skynet_z.fits'
#cat_gold_pofz_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv16/flat_skynet_pofz.fits'
cat_gold_pofz_binned_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv16/flat_skynet_pofz_binned.fits'

shear = 'ngmix'

zmean_type = 1 # using zmean
nofz_type = 0  # using nofz

cat_bench_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/benchmark/benchmark_old_photometry_TPZv1_BPZv2_wavg_tpc_badflag_type.ssv'
mask_bench_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/benchmark/mask_healpix_nest_ns4096_sva1_gold_1.0.2-4_magautoi.ge.22p5_goodregions_04_fracdet.ge.0.8.dat'
cat_gold_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv18/des_sv_wl_info.fits'
cat_gold_mag_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/sva1_gold_1.0.2_catalog_auto.fits'
cat_wl_info_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv18/des_sv_wl_info.fits'
cat_wl_im3shape_name ='/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv18/des_sv_wl_im3shape.fits' 
cat_wl_ngmix_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/flatv18/des_sv_wl_ngmix.fits'
depth_map_name = '/Users/chihwaychang/Desktop/Work/DES_massmap_final/data/sva1_gold_1.0.2-4_nside4096_nest_i_auto_weights.fits'
ran_name = temp_dir_data+'random_gal_data.fits'
photoz_data_file = '/Volumes/astro/refreg/data/des_photoz/PHOTO_Z_GOLD_FINAL2.h5'

