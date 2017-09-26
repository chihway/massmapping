"""
Calculate moments.
"""

import healpy as hp
import pylab as mplot
import numpy as np
import sys
sys.path.append('../utils')
import kmeans_radec
import fitsio

buzzid = int(sys.argv[1])
smoothing = int(sys.argv[2])

shear = 'mcal'
zbin = 6
seed = 100
Njk = 100
sim_dir = 'maps/'
data_dir = 'maps_small/'

k = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_k.fits.gz')[1]['k'][:]
k_alm = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_kalm.fits.gz')[1]['kalm'][:]
g2kE = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_g2kE.fits.gz')[1]['g2kE'][:]
g2kE_ran = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_g2kE_ran.fits.gz')[1]['g2kE_ran'][:]
e2kE = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_e2kE.fits.gz')[1]['e2kE'][:]
e2kE_ran = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_e2kE_ran.fits.gz')[1]['e2kE_ran'][:]
N = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_Nsource.fits.gz')[1]['Nsource'][:]

e2kE_data = fitsio.FITS(data_dir+'y1a1_spt_'+str(shear)+'_'+str(zbin)+'_kE.fits')[1]['kE'][:]
e2kE_ran_data = fitsio.FITS(data_dir+'y1a1_spt_'+str(shear)+'_'+str(zbin)+'_kE_ran.fits')[1]['kE_ran'][:]
N_data = fitsio.FITS(data_dir+'y1a1_spt_'+str(shear)+'_'+str(zbin)+'_Nsource.fits')[1]['Nsource'][:]

mask = N.copy()
mask[mask>0] = 1.0

mask_data = N_data.copy()
mask_data[mask_data>0] = 1.0

mask_all = mask*mask_data
mask = mask_all.copy()
mask_data = mask_all.copy()

k *= mask
g2kE *= mask
g2kE_ran *= mask
e2kE *= mask
e2kE_ran *= mask
e2kE_data *= mask_data
e2kE_ran_data *= mask_data

nside = hp.npix2nside(len(k))
pix = np.arange(hp.nside2npix(nside))
theta, phi = hp.pix2ang(nside, pix)
ra = phi/np.pi*180.0
dec = 90.0 - theta/np.pi*180

Smooth = [0.0, 2., 3.2, 5.1, 8.2, 13.1, 21.0, 33.6]
# 2*1.6**np.arange(7)

if smoothing==1:

    for i in range(8):  
        print "smoothing"

        k_sm = hp.smoothing(k,sigma=Smooth[i]/60/180*np.pi)
        k_alm_sm = hp.smoothing(k_alm,sigma=Smooth[i]/60/180*np.pi)
        g2k_E_sm = hp.smoothing(g2kE,sigma=Smooth[i]/60/180*np.pi)
        g2k_E_ran_sm = hp.smoothing(g2kE_ran,sigma=Smooth[i]/60/180*np.pi)
        e2k_E_sm = hp.smoothing(e2kE,sigma=Smooth[i]/60/180*np.pi)
        e2k_E_ran_sm = hp.smoothing(e2kE_ran,sigma=Smooth[i]/60/180*np.pi)
        mask_sm = hp.smoothing(mask,sigma=Smooth[i]/60/180*np.pi)
        mask_data_sm = hp.smoothing(mask_data,sigma=Smooth[i]/60/180*np.pi)

        e2k_E_data_sm = hp.smoothing(e2kE_data,sigma=Smooth[i]/60/180*np.pi)
        e2k_E_ran_data_sm = hp.smoothing(e2kE_ran_data,sigma=Smooth[i]/60/180*np.pi)
        mask_sm[mask==0] = 0
        mask_data_sm[mask_data==0] = 0

        # write all the maps
        names = ['mask', 'k', 'kalm', 'g2kE', 'g2kE_ran', 'e2kE', 'e2kE_ran', 'e2kE_data', 'e2kE_data_ran', 'mask_data']
        Maps = [mask_sm, k_sm, k_alm_sm, g2k_E_sm, g2k_E_ran_sm, e2k_E_sm, e2k_E_ran_sm, e2k_E_data_sm, e2k_E_ran_data_sm, mask_data_sm]

        # write all the maps
        for j in range(len(names)):
            if j<=6:
                fits = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_'+str(names[j])+'_'+str(Smooth[i])+'.fits','rw',clobber=True)
            else:
                fits = fitsio.FITS(data_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_'+str(names[j])+'_'+str(Smooth[i])+'.fits','rw',clobber=True)

            output = np.zeros(hp.nside2npix(nside), dtype=[(names[j],'f8')])
            output[names[j]] = Maps[j]
            fits.write(output)

        print "making JK's"

        mask_temp = (mask_sm>0) #*(ra>10)*(ra<90)
        ra_temp = ra[mask_temp]
        dec_temp = dec[mask_temp]
        pix_temp = pix[mask_temp]

        RADEC = np.zeros((len(ra_temp),2))
        RADEC[:,0] = ra_temp
        RADEC[:,1] = dec_temp
        dilute_ids = np.arange(len(ra_temp))
        np.random.seed(seed)
        np.random.shuffle(dilute_ids)
        dilute_ids = dilute_ids[:len(ra_temp)/10]
        km = kmeans_radec.kmeans_sample(RADEC[dilute_ids], Njk, maxiter=500, tol=1e-05)
        JK = np.zeros(hp.nside2npix(nside))
        JK[pix_temp] = km.find_nearest(RADEC)+1

        fits = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_mask_jk_'+str(Smooth[i])+'.fits','rw',clobber=True)
        output = np.zeros(hp.nside2npix(nside), dtype=[('JK','f8')])
        output['JK'] = JK
        fits.write(output)

        mask_temp = (mask_data_sm>0) #*(ra>10)*(ra<90)
        ra_temp = ra[mask_temp]
        dec_temp = dec[mask_temp]
        pix_temp = pix[mask_temp]

        RADEC = np.zeros((len(ra_temp),2))
        RADEC[:,0] = ra_temp
        RADEC[:,1] = dec_temp
        dilute_ids = np.arange(len(ra_temp))
        np.random.seed(seed)
        np.random.shuffle(dilute_ids)
        dilute_ids = dilute_ids[:len(ra_temp)/10] # 10
        km = kmeans_radec.kmeans_sample(RADEC[dilute_ids], Njk, maxiter=500, tol=1e-05)
        JK = np.zeros(hp.nside2npix(nside))
        JK[pix_temp] = km.find_nearest(RADEC)+1

        fits = fitsio.FITS(data_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_mask_data_jk_'+str(Smooth[i])+'.fits','rw',clobber=True)
        output = np.zeros(hp.nside2npix(nside), dtype=[('JK','f8')])
        output['JK'] = JK
        fits.write(output)

if smoothing==0:

    KK_true = []
    KK_alm = []
    g2KK_E = []
    g2KK_E_denoise = []
    e2KK_E = []
    e2KK_E_denoise = []
    e2KK_E_data = []
    e2KK_E_denoise_data = []

    KK_true_sig = []
    KK_alm_sig = []
    g2KK_E_sig = []
    g2KK_E_denoise_sig = []
    e2KK_E_sig = []
    e2KK_E_denoise_sig = []
    e2KK_E_data_sig = []
    e2KK_E_denoise_data_sig = []

    # 3rd
    KK_true3 = []
    KK_alm3 = []
    g2KK_E_denoise3 = []
    e2KK_E_denoise3 = []
    e2KK_E_denoise_data3 = []

    KK_true3_sig = []
    KK_alm3_sig = []
    g2KK_E_denoise3_sig = []
    e2KK_E_denoise3_sig = []
    e2KK_E_denoise_data3_sig = []

    # 4th
    KK_true4 = []
    KK_alm4 = []
    g2KK_E_denoise4 = []
    e2KK_E_denoise4 = []
    e2KK_E_denoise_data4 = []

    KK_true4_sig = []
    KK_alm4_sig = []
    g2KK_E_denoise4_sig = []
    e2KK_E_denoise4_sig = []
    e2KK_E_denoise_data4_sig = []

    # 5th
    KK_true5 = []
    KK_alm5 = []
    g2KK_E_denoise5 = []
    e2KK_E_denoise5 = []
    e2KK_E_denoise_data5 = []

    KK_true5_sig = []
    KK_alm5_sig = []
    g2KK_E_denoise5_sig = []
    e2KK_E_denoise5_sig = []
    e2KK_E_denoise_data5_sig = []

    for i in range(8):  
    
        print "calculate moments"

        mask_sm =  fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_mask_'+str(Smooth[i])+'.fits')[1]['mask'][:]
        JK =  fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_mask_jk_'+str(Smooth[i])+'.fits')[1]['JK'][:]
        k_sm =  fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_k_'+str(Smooth[i])+'.fits')[1]['k'][:]
        k_alm_sm = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_kalm_'+str(Smooth[i])+'.fits')[1]['kalm'][:]
        g2k_E_sm = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_g2kE_'+str(Smooth[i])+'.fits')[1]['g2kE'][:]
        g2k_E_ran_sm = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_g2kE_ran_'+str(Smooth[i])+'.fits')[1]['g2kE_ran'][:]
        e2k_E_sm = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_e2kE_'+str(Smooth[i])+'.fits')[1]['e2kE'][:]
        e2k_E_ran_sm = fitsio.FITS(sim_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_e2kE_ran_'+str(Smooth[i])+'.fits')[1]['e2kE_ran'][:]
        e2k_E_data_sm = fitsio.FITS(data_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_e2kE_data_'+str(Smooth[i])+'.fits')[1]['e2kE_data'][:]
        e2k_E_ran_data_sm = fitsio.FITS(data_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_e2kE_data_ran_'+str(Smooth[i])+'.fits')[1]['e2kE_data_ran'][:]
        mask_data_sm =  fitsio.FITS(data_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_mask_data_'+str(Smooth[i])+'.fits')[1]['mask_data'][:]
        JK_data =  fitsio.FITS(data_dir+'buzzard_y1_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'_mask_data_jk_'+str(Smooth[i])+'.fits')[1]['JK'][:]

        KK_true_temp = []
        KK_alm_temp = []
        g2KK_E_temp = []
        g2KK_E_denoise_temp = []
        e2KK_E_temp = []
        e2KK_E_denoise_temp = []
        e2KK_E_data_temp = []
        e2KK_E_denoise_data_temp = []
    
        # 3rd
        KK_true_temp3 = []
        KK_alm_temp3 = []
        g2KK_E_denoise_temp3 = []
        e2KK_E_denoise_temp3 = []
        e2KK_E_denoise_data_temp3 = []

        # 4th
        KK_true_temp4 = []
        KK_alm_temp4 = []
        g2KK_E_denoise_temp4 = []
        e2KK_E_denoise_temp4 = []
        e2KK_E_denoise_data_temp4 = []

        # 5th
        KK_true_temp5 = []
        KK_alm_temp5 = []
        g2KK_E_denoise_temp5 = []
        e2KK_E_denoise_temp5 = []
        e2KK_E_denoise_data_temp5 = []

        for j in range(Njk): 
            print j
            jk_mask = (JK!=j+1)*(JK!=0)
            jk_mask_data = (JK_data!=j+1)*(JK_data!=0)

            k_pix_sm = k_sm[jk_mask]/mask_sm[jk_mask]
            k_alm_pix_sm = k_alm_sm[jk_mask]/mask_sm[jk_mask]
            g2kE_pix_sm = g2k_E_sm[jk_mask]/mask_sm[jk_mask]
            g2kE_ran_pix_sm = g2k_E_ran_sm[jk_mask]/mask_sm[jk_mask]
            e2kE_pix_sm = e2k_E_sm[jk_mask]/mask_sm[jk_mask]
            e2kE_ran_pix_sm = e2k_E_ran_sm[jk_mask]/mask_sm[jk_mask]
            e2kE_data_pix_sm = e2k_E_data_sm[jk_mask_data]/mask_data_sm[jk_mask_data]
            e2kE_ran_data_pix_sm = e2k_E_ran_data_sm[jk_mask_data]/mask_data_sm[jk_mask_data]

            k_pix_sm -= np.mean(k_pix_sm)
            k_alm_pix_sm -= np.mean(k_alm_pix_sm)        
            g2kE_pix_sm -= np.mean(g2kE_pix_sm)
            g2kE_ran_pix_sm -= np.mean(g2kE_ran_pix_sm)
            e2kE_pix_sm -= np.mean(e2kE_pix_sm)
            e2kE_ran_pix_sm -= np.mean(e2kE_ran_pix_sm)
            e2kE_data_pix_sm -= np.mean(e2kE_data_pix_sm)
            e2kE_ran_data_pix_sm -= np.mean(e2kE_ran_data_pix_sm)

            KK_true_temp.append(np.mean(k_pix_sm**2))
            KK_alm_temp.append(np.mean(k_alm_pix_sm**2))
            g2KK_E_temp.append(np.mean(g2kE_pix_sm**2))
            g2KK_E_denoise_temp.append(np.mean(g2kE_pix_sm**2) - np.mean(g2kE_ran_pix_sm**2))
            e2KK_E_temp.append(np.mean(e2kE_pix_sm**2))
            e2KK_E_denoise_temp.append(np.mean(e2kE_pix_sm**2) - np.mean(e2kE_ran_pix_sm**2))
            e2KK_E_data_temp.append(np.mean(e2kE_data_pix_sm**2))
            e2KK_E_denoise_data_temp.append(np.mean(e2kE_data_pix_sm**2) - np.mean(e2kE_ran_data_pix_sm**2))
    
            # 3rd
            KK_true_temp3.append(np.mean(k_pix_sm**3))
            KK_alm_temp3.append(np.mean(k_alm_pix_sm**3))
            g2KK_E_denoise_temp3.append(np.mean(g2kE_pix_sm**3))
            e2KK_E_denoise_temp3.append(np.mean(e2kE_pix_sm**3))
            e2KK_E_denoise_data_temp3.append(np.mean(e2kE_data_pix_sm**3))

            # 4th
            KK_true_temp4.append(np.mean(k_pix_sm**4))
            KK_alm_temp4.append(np.mean(k_alm_pix_sm**4))
            g2KK_E_denoise_temp4.append(np.mean(g2kE_pix_sm**4)-6*np.mean(g2kE_pix_sm**2*g2kE_ran_pix_sm**2)-np.mean(g2kE_ran_pix_sm**4))
            e2KK_E_denoise_temp4.append(np.mean(e2kE_pix_sm**4)-6*np.mean(e2kE_pix_sm**2*e2kE_ran_pix_sm**2)-np.mean(e2kE_ran_pix_sm**4))
            e2KK_E_denoise_data_temp4.append(np.mean(e2kE_data_pix_sm**4)-6*np.mean(e2kE_data_pix_sm**2*e2kE_ran_data_pix_sm**2)-np.mean(e2kE_ran_data_pix_sm**4))

            # 5th
            KK_true_temp5.append(np.mean(k_pix_sm**5))
            KK_alm_temp5.append(np.mean(k_alm_pix_sm**5))
            g2KK_E_denoise_temp5.append(np.mean(g2kE_pix_sm**5)-10*np.mean(g2kE_pix_sm**3*g2kE_ran_pix_sm**2))
            e2KK_E_denoise_temp5.append(np.mean(e2kE_pix_sm**5)-10*np.mean(e2kE_pix_sm**3*e2kE_ran_pix_sm**2))
            e2KK_E_denoise_data_temp5.append(np.mean(e2kE_data_pix_sm**5)-10*np.mean(e2kE_data_pix_sm**3*e2kE_ran_data_pix_sm**2))


        KK_true_temp = np.array(KK_true_temp)
        KK_alm_temp = np.array(KK_alm_temp)
        g2KK_E_temp = np.array(g2KK_E_temp)
        g2KK_E_denoise_temp = np.array(g2KK_E_denoise_temp)
        e2KK_E_temp = np.array(e2KK_E_temp)
        e2KK_E_denoise_temp = np.array(e2KK_E_denoise_temp)
        e2KK_E_data_temp = np.array(e2KK_E_data_temp)
        e2KK_E_denoise_data_temp = np.array(e2KK_E_denoise_data_temp)


        # 3rd
        KK_true_temp3 = np.array(KK_true_temp3)
        KK_alm_temp3 = np.array(KK_alm_temp3)
        g2KK_E_denoise_temp3 = np.array(g2KK_E_denoise_temp3)
        e2KK_E_denoise_temp3 = np.array(e2KK_E_denoise_temp3)
        e2KK_E_denoise_data_temp3 = np.array(e2KK_E_denoise_data_temp3)

        #np.savez('3rd_moment_'+str(i)+'.npz', KK_true=KK_true_temp3, KK_alm=KK_alm_temp3, 
        #g2KK_E_denoise=g2KK_E_denoise_temp3, e2KK_E_denoise=e2KK_E_denoise_temp3, e2KK_E_denoise_data=e2KK_E_denoise_data_temp3)

        # 4th
        KK_true_temp4 = np.array(KK_true_temp4)
        KK_alm_temp4 = np.array(KK_alm_temp4)
        g2KK_E_denoise_temp4 = np.array(g2KK_E_denoise_temp4)
        e2KK_E_denoise_temp4 = np.array(e2KK_E_denoise_temp4)
        e2KK_E_denoise_data_temp4 = np.array(e2KK_E_denoise_data_temp4)

        # 5th
        KK_true_temp5 = np.array(KK_true_temp5)
        KK_alm_temp5 = np.array(KK_alm_temp5)
        g2KK_E_denoise_temp5 = np.array(g2KK_E_denoise_temp5)
        e2KK_E_denoise_temp5 = np.array(e2KK_E_denoise_temp5)
        e2KK_E_denoise_data_temp5 = np.array(e2KK_E_denoise_data_temp5)

        KK_true.append(np.mean(KK_true_temp))
        KK_alm.append(np.mean(KK_alm_temp))
        g2KK_E.append(np.mean(g2KK_E_temp))
        g2KK_E_denoise.append(np.mean(g2KK_E_denoise_temp))
        e2KK_E.append(np.mean(e2KK_E_temp))
        e2KK_E_denoise.append(np.mean(e2KK_E_denoise_temp))
        e2KK_E_data.append(np.mean(e2KK_E_data_temp))
        e2KK_E_denoise_data.append(np.mean(e2KK_E_denoise_data_temp))

        KK_true_sig.append(np.sum((KK_true_temp-KK_true[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        KK_alm_sig.append(np.sum((KK_alm_temp-KK_alm[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        g2KK_E_sig.append(np.sum((g2KK_E_temp-g2KK_E[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        g2KK_E_denoise_sig.append(np.sum((g2KK_E_denoise_temp-g2KK_E_denoise[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        e2KK_E_sig.append(np.sum((e2KK_E_temp-e2KK_E[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        e2KK_E_denoise_sig.append(np.sum((e2KK_E_denoise_temp-e2KK_E_denoise[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        e2KK_E_data_sig.append(np.sum((e2KK_E_data_temp-e2KK_E_data[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        e2KK_E_denoise_data_sig.append(np.sum((e2KK_E_denoise_data_temp-e2KK_E_denoise_data[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)


        # 3rd
        KK_true3.append(np.mean(KK_true_temp3))
        KK_alm3.append(np.mean(KK_alm_temp3))
        g2KK_E_denoise3.append(np.mean(g2KK_E_denoise_temp3))
        e2KK_E_denoise3.append(np.mean(e2KK_E_denoise_temp3))
        e2KK_E_denoise_data3.append(np.mean(e2KK_E_denoise_data_temp3))

        KK_true3_sig.append(np.sum((KK_true_temp3-KK_true3[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        KK_alm3_sig.append(np.sum((KK_alm_temp3-KK_alm3[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        g2KK_E_denoise3_sig.append(np.sum((g2KK_E_denoise_temp3-g2KK_E_denoise3[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        e2KK_E_denoise3_sig.append(np.sum((e2KK_E_denoise_temp3-e2KK_E_denoise3[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        e2KK_E_denoise_data3_sig.append(np.sum((e2KK_E_denoise_data_temp3-e2KK_E_denoise_data3[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)

        # 4th
        KK_true4.append(np.mean(KK_true_temp4))
        KK_alm4.append(np.mean(KK_alm_temp4))
        g2KK_E_denoise4.append(np.mean(g2KK_E_denoise_temp4))
        e2KK_E_denoise4.append(np.mean(e2KK_E_denoise_temp4))
        e2KK_E_denoise_data4.append(np.mean(e2KK_E_denoise_data_temp4))

        KK_true4_sig.append(np.sum((KK_true_temp4-KK_true4[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        KK_alm4_sig.append(np.sum((KK_alm_temp4-KK_alm4[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        g2KK_E_denoise4_sig.append(np.sum((g2KK_E_denoise_temp4-g2KK_E_denoise4[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        e2KK_E_denoise4_sig.append(np.sum((e2KK_E_denoise_temp4-e2KK_E_denoise4[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        e2KK_E_denoise_data4_sig.append(np.sum((e2KK_E_denoise_data_temp4-e2KK_E_denoise_data4[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)

        # 5th
        KK_true5.append(np.mean(KK_true_temp5))
        KK_alm5.append(np.mean(KK_alm_temp5))
        g2KK_E_denoise5.append(np.mean(g2KK_E_denoise_temp5))
        e2KK_E_denoise5.append(np.mean(e2KK_E_denoise_temp5))
        e2KK_E_denoise_data5.append(np.mean(e2KK_E_denoise_data_temp5))

        KK_true5_sig.append(np.sum((KK_true_temp5-KK_true5[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        KK_alm5_sig.append(np.sum((KK_alm_temp5-KK_alm5[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        g2KK_E_denoise5_sig.append(np.sum((g2KK_E_denoise_temp5-g2KK_E_denoise5[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        e2KK_E_denoise5_sig.append(np.sum((e2KK_E_denoise_temp5-e2KK_E_denoise5[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)
        e2KK_E_denoise_data5_sig.append(np.sum((e2KK_E_denoise_data_temp5-e2KK_E_denoise_data5[-1])**2)**0.5/Njk**0.5*(Njk-1)**0.5)


    np.savez('kappa_moments_'+str(shear)+'_'+str(buzzid)+'_'+str(zbin)+'.npz', 
    KK_true = np.array(KK_true),
    KK_alm = np.array(KK_alm),
    g2KK_E = np.array(g2KK_E),
    g2KK_E_denoise = np.array(g2KK_E_denoise),
    e2KK_E = np.array(e2KK_E),
    e2KK_E_denoise = np.array(e2KK_E_denoise),
    e2KK_E_data = np.array(e2KK_E_data),
    e2KK_E_denoise_data = np.array(e2KK_E_denoise_data),

    KK_true_sig = np.array(KK_true_sig),
    KK_alm_sig = np.array(KK_alm_sig),
    g2KK_E_sig = np.array(g2KK_E_sig),
    g2KK_E_denoise_sig = np.array(g2KK_E_denoise_sig),
    e2KK_E_sig = np.array(e2KK_E_sig),
    e2KK_E_denoise_sig = np.array(e2KK_E_denoise_sig),
    e2KK_E_data_sig = np.array(e2KK_E_data_sig),
    e2KK_E_denoise_data_sig = np.array(e2KK_E_denoise_data_sig),

    KK_true3 = np.array(KK_true3),
    KK_alm3 = np.array(KK_alm3),
    g2KK_E_denoise3 = np.array(g2KK_E_denoise3),
    e2KK_E_denoise3 = np.array(e2KK_E_denoise3),
    e2KK_E_denoise_data3 = np.array(e2KK_E_denoise_data3),

    KK_true3_sig = np.array(KK_true3_sig),
    KK_alm3_sig = np.array(KK_alm3_sig),
    g2KK_E_denoise3_sig = np.array(g2KK_E_denoise3_sig),
    e2KK_E_denoise3_sig = np.array(e2KK_E_denoise3_sig),
    e2KK_E_denoise_data3_sig = np.array(e2KK_E_denoise_data3_sig),

    KK_true4 = np.array(KK_true4),
    KK_alm4 = np.array(KK_alm4),
    g2KK_E_denoise4 = np.array(g2KK_E_denoise4),
    e2KK_E_denoise4 = np.array(e2KK_E_denoise4),
    e2KK_E_denoise_data4 = np.array(e2KK_E_denoise_data4),

    KK_true4_sig = np.array(KK_true4_sig),
    KK_alm4_sig = np.array(KK_alm4_sig),
    g2KK_E_denoise4_sig = np.array(g2KK_E_denoise4_sig),
    e2KK_E_denoise4_sig = np.array(e2KK_E_denoise4_sig),
    e2KK_E_denoise_data4_sig = np.array(e2KK_E_denoise_data4_sig),

    KK_true5 = np.array(KK_true5),
    KK_alm5 = np.array(KK_alm5),
    g2KK_E_denoise5 = np.array(g2KK_E_denoise5),
    e2KK_E_denoise5 = np.array(e2KK_E_denoise5),
    e2KK_E_denoise_data5 = np.array(e2KK_E_denoise_data5),

    KK_true5_sig = np.array(KK_true5_sig),
    KK_alm5_sig = np.array(KK_alm5_sig),
    g2KK_E_denoise5_sig = np.array(g2KK_E_denoise5_sig),
    e2KK_E_denoise5_sig = np.array(e2KK_E_denoise5_sig),
    e2KK_E_denoise_data5_sig = np.array(e2KK_E_denoise_data5_sig))


