import numpy as np 
import cosmolopy.distance as cd
from astropy.convolution import Box2DKernel
from astropy.convolution import Tophat2DKernel
from astropy import convolution


def w_kernel(z_l, z_s, cosmo):
    """
    [script from Arnau]
    It defines the lensing kernel for a given comoving distance of 
    lens and source. 
    """
    a = 1.0/(1 + z_l)
    cd_l = cd.comoving_distance(z_l, **cosmo)
    cd_s = cd.comoving_distance(z_s, **cosmo)
    return ((cd_l * (cd_s-cd_l)/cd_s))/a

def w_kernel2(z_l, z_s, cosmo):
    """
    [script from Arnau]
    It defines the lensing kernel for a given comoving distance of 
    lens and source. 
    """
    a = 1.0/(1 + z_l)
    cd_l = cd.comoving_distance(z_l, **cosmo)
    cd_s = cd.comoving_distance(z_s, **cosmo)
    return (cd_s-cd_l)/cd_s

def lens_weight(zbins_l, mean_z_s, cosmo):
    """
    [script from Arnau]
    It calculates the mean lensing kernel in a given lensing distance 
    binning. 
    """
    w = np.zeros(len(zbins_l) - 1)
    for i in range(len(w)):
        z_subbins = np.linspace(zbins_l[i], zbins_l[i + 1], 100)
        w[i] = np.sum(w_kernel(z_subbins, mean_z_s, cosmo))/100.
    return w

def lens_weight_nofz(nofz_l, nofz_s, cosmo):
    """
    It calculates the mean lensing kernel given nofz of lens and source bin. 
    """
    nofz_l = nofz_l/np.sum(nofz_l)
    nofz_s = nofz_s/np.sum(nofz_s)

    
    z_subbins = np.linspace(0, 1.8, 200)
    dz_cd = cd.comoving_distance(z_subbins[1:], **cosmo) - cd.comoving_distance(z_subbins[:-1], **cosmo)
    
    w = 0
    for i in range(200-1):
       
        # try other way around
        mask = (z_subbins>z_subbins[i])
        if len(z_subbins[mask])>1:
            #print(len(z_subbins[mask]), len(dz_cd), len(mask))
            w += np.sum(w_kernel2(z_subbins[i], z_subbins[mask], cosmo)*nofz_s[mask]*dz_cd[mask[1:]])*cd.comoving_distance(z_subbins[i], **cosmo)*nofz_l[i]/(1.0/(1 + z_subbins[i]))*dz_cd[i]
    return w

def lens_weight_nofz2(z_l1, z_l2, nofz_s, cosmo):
    """
    It calculates the mean lensing kernel given nofz of lens and source bin. 
    """
    nofz_s = nofz_s/np.sum(nofz_s)
    
    z_subbins = np.linspace(0, 1.8, 200)
    nofz_l = np.ones(200)*1.0
    mask = (z_subbins<=z_l1)|(z_subbins>z_l2)
    nofz_l[mask] = 0
    nofz_l = nofz_l/np.sum(nofz_l)
    dz_cd = cd.comoving_distance(z_subbins[1:], **cosmo) - cd.comoving_distance(z_subbins[:-1], **cosmo)
    
    w = 0
    for i in range(200-1):
        
        # try other way around
        mask = (z_subbins>z_subbins[i])
        if len(z_subbins[mask])>1:
            w += np.sum(w_kernel2(z_subbins[i], z_subbins[mask], cosmo)*nofz_s[mask]*dz_cd[mask[1:]])*cd.comoving_distance(z_subbins[i], **cosmo)*nofz_l[i]/(1.0/(1 + z_subbins[i]))*dz_cd[i]
    return w


def kappag_map_bin(N2d, mask, fraction, z_l1, z_l2, z_s, cosmo, smooth_kernal, smooth_scale):
    """
    Calculate kappa_g map for a lens redshift bin from z_l1 to z_l2 
    and a source redshift of z_s.
    """

    c_light = 3.0e5
    if smooth_kernal=='box':
        kernel = Box2DKernel(smooth_scale)
    if smooth_kernal=='tophat':
        kernel = Tophat2DKernel(smooth_scale/2)

    # make 2D galaxy over-density maps ########
    if fraction==None:
        fraction = mask.copy()


    N2d = N2d*mask

    area_fraction = np.sum(fraction[mask==1])/len(fraction[mask==1])

    dN2d = N2d*0.0
    ave = np.mean(N2d[mask==1])/area_fraction
    dN2d[mask==1] = (N2d[mask==1]-(ave*fraction[mask==1]))/(ave*fraction[mask==1])
    # print(ave)

    # make 2D kappa_g maps ########

    zl1_cd = cd.comoving_distance(z_l1, **cosmo) # Mpc
    zl2_cd = cd.comoving_distance(z_l2, **cosmo) # Mpc 
    zs_cd = cd.comoving_distance(z_s, **cosmo) # Mpc 
    delta_cd = zl2_cd - zl1_cd
    const = ((100. * cosmo['h'])**2 * cosmo['omega_M_0']) * (3/2.) * (1/c_light**2)     

    integ = lens_weight(np.array([z_l1, z_l2]), z_s, cosmo)[0]

    temp_dN = dN2d*1.0            
    temp_dN[mask==0]='nan'            
    kg = const * delta_cd * integ * convolution.convolve_fft(temp_dN, kernel, normalize_kernel=True, ignore_edge_zeros=True, interpolate_nan=True)
    kg[mask==0]=0
    return kg

def kappag_map_bin_nofz(N2d, mask, nofz_l, nofz_s, cosmo, smooth_kernal, smooth_scale):
    """
    Calculate kappa_g map for a lens redshift bin with nofz_l and a 
    source redshift with nofz_s.

    nofz_l and nofz_s are arrays that are normalized to 1 and with 200 bins from 0 to 1.8.
    """

    c_light = 3.0e5
    if smooth_kernal=='box':
        kernel = Box2DKernel(smooth_scale)
    if smooth_kernal=='tophat':
        kernel = Tophat2DKernel(smooth_scale/2)

    # make 2D galaxy over-density maps ########

    dN2d = N2d*0.0
    ave = np.mean(N2d[mask==1])
    dN2d[mask==1] = (N2d[mask==1]-ave)/ave

    # make 2D kappa_g maps ########
    const = ((100. * cosmo['h'])**2 * cosmo['omega_M_0']) * (3/2.) * (1/c_light**2)
   
    integ = lens_weight_nofz(nofz_l, nofz_s, cosmo)
    temp_dN = dN2d*1.0            
    temp_dN[mask==0]='nan'            
    
    kg = const * integ * convolution.convolve_fft(temp_dN, kernel, normalize_kernel=True, ignore_edge_zeros=True, interpolate_nan=True)
    
    kg[mask==0]=0
    return kg

def kappag_map_bin_nofz2(N2d, mask, z_l1, z_l2, nofz_s, cosmo, smooth_kernal, smooth_scale):
    """
    Calculate kappa_g map for a lens redshift bin with nofz_l and a 
    source redshift with nofz_s.

    nofz_s are arrays that are normalized to 1 and with 200 bins from 0 to 1.8.
    """

    c_light = 3.0e5
    if smooth_kernal=='box':
        kernel = Box2DKernel(smooth_scale)
    if smooth_kernal=='tophat':
        kernel = Tophat2DKernel(smooth_scale/2)

    # make 2D galaxy over-density maps ########

    dN2d = N2d*0.0
    ave = np.mean(N2d[mask==1])
    dN2d[mask==1] = (N2d[mask==1]-ave)/ave

    # make 2D kappa_g maps ########
    const = ((100. * cosmo['h'])**2 * cosmo['omega_M_0']) * (3/2.) * (1/c_light**2)
   
    integ = lens_weight_nofz2(z_l1, z_l2, nofz_s, cosmo)
    temp_dN = dN2d*1.0            
    temp_dN[mask==0]='nan'            
    
    kg = const * integ * convolution.convolve_fft(temp_dN, kernel, normalize_kernel=True, ignore_edge_zeros=True, interpolate_nan=True)
    
    kg[mask==0]=0
    return kg
