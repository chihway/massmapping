
import numpy as np
import astropy.io.fits as pf
import healpy as hp
from scipy import fftpack
from astropy import convolution
from astropy.convolution import Box2DKernel
from astropy.convolution import Tophat2DKernel

### Stuff used in Y1

def make_healpix_map(ra, dec, val, w, nside=1024):
    """
    Pixelate catalog into Healpix maps.
    """

    theta = (90.0 - dec)*np.pi/180.
    phi = ra*np.pi/180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    map_count = np.zeros(hp.nside2npix(nside))
    map_countw = np.zeros(hp.nside2npix(nside))
    map_val = np.zeros(hp.nside2npix(nside))

    for i in xrange(len(pix)):
        map_count[pix[i]] += 1
        map_countw[pix[i]] += w[i]
        map_val[pix[i]] += val[i]*w[i]
    return map_count, map_countw, map_val

def make_healpix_map_sim_all(ra, dec, ra_ran, dec_ran, k, g1, g2, e1, e2, w, nside=1024):
    """
    For simulations, make all Healpix maps at once: true kappa, true shear, 
    ellipticity, random shear and ellipticity.
    """

    theta = (90.0 - dec)*np.pi/180.
    phi = ra*np.pi/180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    theta_ran = (90.0 - dec_ran)*np.pi/180.
    phi_ran = ra_ran*np.pi/180.
    pix_ran = hp.ang2pix(nside, theta_ran, phi_ran, nest=False)

    map_count = np.zeros(hp.nside2npix(nside))
    map_countw = np.zeros(hp.nside2npix(nside))
    map_k = np.zeros(hp.nside2npix(nside))
    map_g1 = np.zeros(hp.nside2npix(nside))
    map_g2 = np.zeros(hp.nside2npix(nside))
    map_e1 = np.zeros(hp.nside2npix(nside))
    map_e2 = np.zeros(hp.nside2npix(nside))
    map_g1_ran = np.zeros(hp.nside2npix(nside))
    map_g2_ran = np.zeros(hp.nside2npix(nside))
    map_e1_ran = np.zeros(hp.nside2npix(nside))
    map_e2_ran = np.zeros(hp.nside2npix(nside))

    for i in xrange(len(pix)):
        pp = pix[i]
        pp_ran = pix_ran[i]
        map_count[pp] += 1
        map_countw[pp] += w[i]
        map_k[pp] += k[i]*w[i]
        map_g1[pp] += g1[i]*w[i]
        map_g2[pp] += g2[i]*w[i]
        map_e1[pp] += e1[i]*w[i]
        map_e2[pp] += e2[i]*w[i]
        map_g1_ran[pp_ran] += g1[i]*w[i]
        map_g2_ran[pp_ran] += g2[i]*w[i]
        map_e1_ran[pp_ran] += e1[i]*w[i]
        map_e2_ran[pp_ran] += e2[i]*w[i]

    return map_count, map_countw, map_k, map_g1, map_g2, map_e1, map_e2, map_g1_ran, map_g2_ran, map_e1_ran, map_e2_ran

def g2k_sphere(kappa, gamma1, gamma2, mask, nside=1024, lmax=2048, synfast=False, sm=False):
    """
    Convert shear to convergence on a sphere. In put are all healpix maps.
    """
    
    kappa_mask = kappa*mask
    gamma1_mask = gamma1*mask
    gamma2_mask = gamma2*mask

    KQU_masked_maps = [kappa_mask, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True) # Spin transform!

    if synfast == False:
        ell, emm = hp.Alm.getlm(lmax=lmax)
        almsE = alms[1]*((ell*(ell+1.))/((ell+2.)*(ell-1)))**0.5
        almsB = alms[2]*((ell*(ell+1.))/((ell+2.)*(ell-1)))**0.5
        almsE[ell==0] = 0.0
        almsB[ell==0] = 0.0
        almsE[ell==1] = 0.0
        almsB[ell==1] = 0.0

    else:
        almsE = alms[1]
        almsB = alms[2]

    almssm = [alms[0], almsE, almsB]
    
    if sm==False:
        kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False)
        E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False)
        B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False)
    
    else:
        kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False, sigma=sm)
        E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False, sigma=sm)
        B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False, sigma=sm)
   
    return kappa_mask, kappa_map_alm, E_map, B_map


def g2phialpha_sphere(gamma1, gamma2, mask, nside=1024, lmax=2048):
    """
    Convert shear to phi and alpha on a sphere. In put are all healpix maps.
    """

    gamma1_mask = gamma1*mask
    gamma2_mask = gamma2*mask

    KQU_masked_maps = [gamma1_mask*0.0, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True) # Spin transform!
    
    # phi
    ell, emm = hp.Alm.getlm(lmax=lmax)
    almsE_phi = -2*((ell+2)*(ell+1)*ell*(ell-1))**(-0.5)*alms[1]
    almsB_phi = -2*((ell+2)*(ell+1)*ell*(ell-1))**(-0.5)*alms[2]
    # do we need B-modes here?
    almsE_phi[ell==0] = 0
    almsE_phi[ell==1] = 0
    almsE_phi = np.nan_to_num(almsE_phi)
    almsB_phi[ell==0] = 0
    almsB_phi[ell==1] = 0
    almsB_phi = np.nan_to_num(almsB_phi)
    
    # alpha
    alms_p_alpha = 2*((ell+2)*(ell-1))**(-0.5)*alms[2]  # check this!!!
    alms_m_alpha = 2*((ell+2)*(ell-1))**(-0.5)*alms[1]
    alms_p_alpha[ell==1] = 0
    alms_p_alpha = np.nan_to_num(alms_p_alpha)
    alms_m_alpha[ell==1] = 0
    alms_m_alpha = np.nan_to_num(alms_m_alpha)
    # do we need B-modes here?
    
    PhiE_map = hp.alm2map(almsE_phi, nside=nside, lmax=lmax, pol=False)
    PhiB_map = hp.alm2map(almsB_phi, nside=nside, lmax=lmax, pol=False)
    Alpha_map = hp.alm2map_spin([alms_p_alpha[emm>=0], alms_m_alpha[emm>=0]], nside, 1, lmax)
    
    return PhiE_map, PhiB_map, Alpha_map[0], Alpha_map[1]


### Stuff used in SV

def k2g_fft(kE, kB, dx, pad=True):
    """
    Convert kappa to gamma in Fourier space. If padding is 
    set to True, include the same size of padding as the data 
    on each side, the total grid is 9 times the original.
    """

    if pad:
        kE_temp = np.zeros((len(kE)*3, len(kE[0])*3))
        kB_temp = np.zeros((len(kB)*3, len(kE[0])*3))
        kE_temp[len(kE):len(kE)*2, len(kE[0]):len(kE[0])*2] = kE*1.0
        kB_temp[len(kB):len(kB)*2, len(kB[0]):len(kB[0])*2] = kB*1.0
        kE_3d_ft = fftpack.fft2(kE_temp)
        kB_3d_ft = fftpack.fft2(kB_temp)
    else:
        kE_3d_ft = fftpack.fft2(kE)
        kB_3d_ft = fftpack.fft2(kB)
    
    FF1 = fftpack.fftfreq(len(kE_3d_ft))
    FF2 = fftpack.fftfreq(len(kE_3d_ft[0]))

    dk = 1.0/dx*2*np.pi                     # max delta_k in 1/arcmin
    kx = np.dstack(np.meshgrid(FF2, FF1))[:,:,0]*dk
    ky = np.dstack(np.meshgrid(FF2, FF1))[:,:,1]*dk
    kx2 = kx**2
    ky2 = ky**2
    k2 = kx2 + ky2

    k2[k2==0] = 1e-15
    k2gamma1_ft = kE_3d_ft/k2*(kx2-ky2) - kB_3d_ft/k2*2*(kx*ky)
    k2gamma2_ft = kE_3d_ft/k2*2*(kx*ky) + kB_3d_ft/k2*(kx2-ky2)

    if pad:
        return fftpack.ifft2(k2gamma1_ft).real[len(kE):len(kE)*2, len(kE[0]):len(kE[0])*2], fftpack.ifft2(k2gamma2_ft).real[len(kE):len(kE)*2, len(kE[0]):len(kE[0])*2]
    else:
        return fftpack.ifft2(k2gamma1_ft).real, fftpack.ifft2(k2gamma2_ft).real

def g2k_fft(g1, g2, dx, pad=True):
    """
    Convert gamma to kappa in Fourier space. If padding is 
    set to True, include the same size of padding as the data 
    on each side, the total grid is 9 times the original.
    """

    if pad:
        g1_temp = np.zeros((len(g1)*3, len(g1[0])*3))
        g2_temp = np.zeros((len(g2)*3, len(g2[0])*3))
        g1_temp[len(g1):len(g1)*2, len(g1[0]):len(g1[0])*2] = g1*1.0
        g2_temp[len(g2):len(g2)*2, len(g2[0]):len(g2[0])*2] = g2*1.0
        g1_3d_ft = fftpack.fft2(g1_temp)
        g2_3d_ft = fftpack.fft2(g2_temp)
    else:
        g1_3d_ft = fftpack.fft2(g1)
        g2_3d_ft = fftpack.fft2(g2)
    FF1 = fftpack.fftfreq(len(g1_3d_ft))
    FF2 = fftpack.fftfreq(len(g1_3d_ft[0]))

    dk = 1.0/dx*2*np.pi                     # max delta_k in 1/arcmin
    kx = np.dstack(np.meshgrid(FF2, FF1))[:,:,0]*dk
    ky = np.dstack(np.meshgrid(FF2, FF1))[:,:,1]*dk
    kx2 = kx**2
    ky2 = ky**2
    k2 = kx2 + ky2

    k2[k2==0] = 1e-15
    g2kappaE_ft = g1_3d_ft/k2*(kx2-ky2) + g2_3d_ft/k2*2*(kx*ky)
    g2kappaB_ft = -1*g1_3d_ft/k2*2*(kx*ky) + g2_3d_ft/k2*(kx2-ky2)

    if pad:
        return fftpack.ifft2(g2kappaE_ft).real[len(g1):len(g1)*2, len(g1[0]):len(g1[0])*2], fftpack.ifft2(g2kappaB_ft).real[len(g1):len(g1)*2, len(g1[0]):len(g1[0])*2]
    else:   
        return fftpack.ifft2(g2kappaE_ft).real, fftpack.ifft2(g2kappaB_ft).real


def smooth_map(map, mask, smooth_kernal, smooth_scale, nan_flag):

    map[mask==0] = nan_flag

    if smooth_kernal=='box':
        kernel = Box2DKernel(smooth_scale)
    if smooth_kernal=='tophat':
        kernel = Tophat2DKernel(smooth_scale/2)

    return convolution.convolve_fft(map, kernel, normalize_kernel=True, ignore_edge_zeros=True, interpolate_nan=True) 


