import numpy as np
import sys
import pandas as pd
import healpy as hp
import os
from astropy.table import Table
import matplotlib.pyplot as plt



#**************************************
#               ROUTINES
#**************************************
def compute_area(map, nside):
    map[map > 0] = 1
    area = np.sum(map) * 1.0 * hp.pixelfunc.nside2pixarea(nside, degrees=True)
    return area

def compute_response(block):
    '''
    Compute Response for Metacalibration
    '''

    dgamma = 2. * 0.01
    selection_mask_1p = (block['snr_1p'] > 10.) & ((block['size_1p'] / block['mcal_psf_size']) > 0.5)
    selection_mask_1m = (block['snr_1m'] > 10.) & ((block['size_1m'] / block['mcal_psf_size']) > 0.5)
    selection_mask_2p = (block['snr_2p'] > 10.) & ((block['size_2p'] / block['mcal_psf_size']) > 0.5)
    selection_mask_2m = (block['snr_2m'] > 10.) & ((block['size_2m'] / block['mcal_psf_size']) > 0.5)
    selection_mask = (block['snr'] > 10.) & ((block['size'] / block['mcal_psf_size']) > 0.5)

    R11 = (block['R11'][selection_mask]).mean() + (block.loc[selection_mask_1p, 'e1'].mean() - block.loc[
        selection_mask_1m, 'e1'].mean()) / dgamma
    R12 = (block['R12'][selection_mask]).mean()
    R22 = (block['R22'][selection_mask]).mean() + (block.loc[selection_mask_2p, 'e2'].mean() - block.loc[
        selection_mask_2m, 'e2'].mean()) / dgamma
    R21 = (block['R21'][selection_mask]).mean()

    return R11, R12, R21, R22, selection_mask

def make_map_full_sky(block):

    # selection sample **************************************
    # criteria: 'snr > 10.' &  	size/psf_size > 0.5
    R11, R12, R21, R22, selection_mask = compute_response(block)

    # convert to map ****************************************
    map1, map_w1, map_e1 = make_healpix_map(np.array(block['RA'][selection_mask]), np.array(block['DEC'][selection_mask]), np.array(block['e1'][selection_mask]),
                                            np.ones(len(block['DEC'][selection_mask])),
                                            nside=1024)
    map2, map_w2, map_e2 = make_healpix_map(np.array(block['RA'][selection_mask]), np.array(block['DEC'][selection_mask]), np.array(block['e2'][selection_mask]),
                                            np.ones(len(block['DEC'][selection_mask])),
                                            nside=1024)



    mask_pixels = map_w1 != 0


    map_e1[mask_pixels] = map_e1[mask_pixels] / (map_w1[mask_pixels] * R11)
    map_e2[mask_pixels] = map_e2[mask_pixels] / (map_w2[mask_pixels] * R22)

    survey_mask = np.copy(map1)
    survey_mask[mask_pixels] = 1.

    # flip sign and correct for mean
    mean_e1 = np.mean(np.array(block['e1'][selection_mask])) / R11
    mean_e2 = np.mean(np.array(block['e2'][selection_mask])) / R22
    map_e1 -= mean_e1
    map_e2 -= mean_e2
    map_e2 = map_e2 * (-1)

    # maps using full sky formalism
    kappa_mask, kappa_map_alm, E_map, B_map = g2k_sphere(0.0 * map_e1, map_e1, map_e2, np.ones(len(map_e1)), nside=nside, lmax=lmax)
    return kappa_mask, kappa_map_alm, E_map, B_map, map1, map_e1, map_e2, survey_mask, selection_mask

def make_healpix_map(ra, dec, val, w, nside=1024):
    """
    Pixelate catalog into Healpix maps.
    """

    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    map_count = np.zeros(hp.nside2npix(nside))
    map_countw = np.zeros(hp.nside2npix(nside))
    map_val = np.zeros(hp.nside2npix(nside))

    for i in range(len(pix)):
        map_count[pix[i]] += 1
        map_countw[pix[i]] += w[i]
        map_val[pix[i]] += val[i] * w[i]
    return map_count, map_countw, map_val

def g2k_sphere(kappa, gamma1, gamma2, mask, nside=1024, lmax=2048, synfast=False, sm=False):
    """
    Convert shear to convergence on a sphere. In put are all healpix maps.
    """

    kappa_mask = kappa * mask
    gamma1_mask = gamma1 * mask
    gamma2_mask = gamma2 * mask

    KQU_masked_maps = [kappa_mask, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)  # Spin transform!

    if synfast == False:  # 1/2 missing? **
        ell, emm = hp.Alm.getlm(lmax=lmax)
        almsE = alms[1] * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
        almsB = alms[2] * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
        almsE[ell == 0] = 0.0
        almsB[ell == 0] = 0.0
        almsE[ell == 1] = 0.0
        almsB[ell == 1] = 0.0

    else:
        almsE = alms[1]
        almsB = alms[2]

    almssm = [alms[0], almsE, almsB]

    if sm == False:
        kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False)
        E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False)
        B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False)

    else:
        kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False, sigma=sm)
        E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False, sigma=sm)
        B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False, sigma=sm)

    return kappa_mask, kappa_map_alm, E_map, B_map

def update_progress(progress,elapsed_time=0,starting_time=0):
    import time

    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))

    if progress*100>1. and elapsed_time>0 :
        remaining=((elapsed_time-starting_time)/progress)*(1.-progress)
        text = "\rPercent: [{0}] {1:.2f}% {2}  - elapsed time: {3} - estimated remaining time: {4}".format( "#"*block + "-"*(barLength-block), progress*100, status,time.strftime('%H:%M:%S',time.gmtime(elapsed_time-starting_time)),time.strftime('%H:%M:%S',time.gmtime(remaining)))
    else:
        text = "\rPercent: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

if __name__ == "__main__":

    # input params ******************************************
    nside = 1024  # 3 arcmin pixels
    lmax = 2 * nside

    input_catalog = './input_files/MOF_matched_Y3_001-0--combined-blind-v1.csv'
    gal_chunk = 20000000

    z_column = 'BPZ_Z_MEAN'
    z_MC_column = 'BPZ_Z_MC'

    ramin = 0.
    ramax = 100.
    decmin = -70.
    decmax = -10.

    name_output = 'test_map_bigger'

    zmin = np.float(input('Insert zmin: '))
    zmax = np.float(input('Insert zmax: '))




    # read the catalog **************************************

    with open(input_catalog, 'r') as galaxy_shapes:
            galaxy_sample_reader = pd.read_csv(galaxy_shapes,
                                           chunksize=gal_chunk,
                                           index_col='COADD_OBJECT_ID'
                                           )
            print('Reading file...')
            count = 0
            for chunk, block in enumerate(galaxy_sample_reader):
                count += gal_chunk
                print ('chunk : {0}, total galaxies loaded : {1}'.format(chunk+1, count))
                # redshift, ra, dec mask *********************************
                mask = (block[z_column] > zmin) & (block[z_column] < zmax) & (block['RA'] < ramax) & (
                block['RA'] >= ramin) & (
                           block['DEC'] < decmax) & (block['DEC'] >= decmin)
                block = block[mask]
                if chunk == 0:
                    block_tot = block
                else:
                    block_tot = block.append(block_tot)

            print('Total galaxies selected: {0}'.format(len(block_tot)))
            # compute k maps ****************************************
            print ('Compute maps')
            kappa_mask, kappa_map_alm, E_map, B_map, map1, map_e1, map_e2, survey_mask, selection_mask = make_map_full_sky(block_tot)


            # save maps and print stats ******************************
            Z = np.array(block_tot[z_MC_column][selection_mask])
            mean_z = np.mean(Z)

            print('Number of sources: {0} ; mean redshift: {1}'.format(len(Z),mean_z))
            plt.hist(Z,bins=np.linspace(0., 1.5, 20))
            plt.savefig('./output_files/'+name_output+'_'+str(zmin)+'_'+str(zmax)+'_redshift_distribution.png')

            path = './output_files/'+name_output+'_MASK_' + str(zmin) + '_' + str(zmax) + '.fits'


            fits_f = Table()
            if not os.path.exists(path):
                fits_f['mask'] = survey_mask
                fits_f.write(path)


            area = compute_area(survey_mask, nside)
            print(('Total area of the sample: {0} deg2').format(area))

            sigma_e1 = np.std(np.array(block_tot['e1'][selection_mask]))
            sigma_e2 = np.std(np.array(block_tot['e2'][selection_mask]))

            print ('std e1 = {0}, std e2 = {1}'.format(sigma_e1, sigma_e2))
            np.savez('./output_files/'+name_output+'_' + str(zmin) + '_' + str(zmax) + '_info.npz',
                     ngal=len(Z), e1_std=sigma_e1, e2_std=sigma_e2, mean_z=mean_z, area=area)

            names = ['kE', 'kB', 'Nsource', 'E1', 'E2']
            Maps = [E_map, B_map, map1, map_e1, map_e2]

            path = './output_files/'+name_output+'_' + str(zmin) + '_' + str(zmax) + '.fits'

            fits_f = Table()
            if not os.path.exists(path):
                for i in range(len(names)):
                    fits_f[names[i]] = Maps[i]
                fits_f.write(path)

