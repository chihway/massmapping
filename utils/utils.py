
import numpy as np
from scipy import stats
from scipy import interpolate
import pylab as mplot
import scipy.optimize as optimization
import healpy as hp
from numpy.linalg import inv

def smooth_map_boxcar(map, size, Nx, Ny, fill):
	"""
	for a 1-0 map, smooth
	"""
	map_smoothed = map*0.0
	Nmap_smoothed = map*0.0
	map_patched = np.zeros((Ny+(size-1), Nx+(size-1))) + fill
	map_patched[(size-1)/2:(size-1)/2+Ny, (size-1)/2:(size-1)/2+Nx] = map
	Nmap_patched = map_patched*1.0
	Nmap_patched[Nmap_patched>0] = 1.0

	for i in range(Ny):
		for j in range(Nx):
			sub_arr = map_patched[i:i+(size-1)+1, j:j+(size-1)+1]
			map_smoothed[i][j] = np.mean(sub_arr)
	return map_smoothed

def invert_mask(mask):
	"""
	invert a 1-0 mask
	"""
	mask[mask==1] = -100
	mask[mask==0] = 1
	mask[mask==-100] = 0
	return mask

def grow_mask(mask, Nbin_ra, Nbin_dec, fill=3, edge=50., pix=5.):
	"""
	grow mask via boxcar smoothing
	"""
	mask_fill = smooth_map_boxcar(mask, fill, Nbin_ra, Nbin_dec, 0) 
	mask_fill[mask_fill>0] = 1
	# invert
	mask_fill_invert = invert_mask(mask_fill)
	mask_fill_invert_grow = smooth_map_boxcar(mask_fill_invert, round(edge/pix)+3, Nbin_ra, Nbin_dec, 1) 
	# invert again
	mask_fill_invert_grow_invert = invert_mask(mask_fill_invert_grow)

	return mask_fill_invert_grow_invert

def get_random_from_hist(NN, Ngal, bin=500):
	"""
	input histogram, output a random draw from that distribution with 
	assigned number of draws.
	"""

	value = (NN[1][1:]+NN[1][:-1])/2
	n = NN[0]
	spl = interpolate.splrep(value, n) 
	value_finer = np.linspace(value.min(), value.max(), bin) # finer bins
	n_interpolate = interpolate.splev(value_finer, spl) # n at finer bins
	n_interpolate /= n_interpolate.sum() # normalizing n
	value_id = np.arange(value_finer.shape[0]) # the bin numbers
	X = stats.rv_discrete(name='random dist', values=(value_id, n_interpolate)) # generating n(value) distribution 
	r = X.rvs(size=Ngal) # generating Ngal random ids
	re = value_finer[r] # the corresponding random values  
	return re

# def w_mean(x, xerr):
# 	"""
# 	return weighted mean.
# 	"""
# 	if len(x)==1:
# 		return x
# 	else:
# 		std = w_std(xerr)
# 		W = np.ones(len(xerr))
# 		Cinv = 1./np.cov(xerr.T)
# 		#Cinv = 1./(np.dot(np.dot(W.T,np.cov(xerr.T)),W))	
# 		return std**2*np.dot(np.dot(W.T, Cinv),x)

# def w_std(xerr):
# 	"""
# 	return error on weighted mean.
# 	"""
	
# 	if len(xerr)==1:
# 		return xerr
	
# 	else:
		
# 		W = np.ones(len(xerr))
# 		# Cinv = 1./(np.dot(np.dot(W.T,np.cov(xerr.T)),W))
# 		Cinv = 1./np.cov(xerr.T)
# 		return (np.dot(np.dot(W.T,Cinv),W))**-0.5

def w_mean_cov(x, xerr):
	"""
	return weighted mean.
	"""
	if len(x)==1:
		return x
	else:
		model = np.ones(len(x))
		W = 1./(np.cov(xerr.T))
		return 1./np.dot(np.dot(model.T, W), model)*np.dot(np.dot(model.T, W), x)

def w_std_cov(xerr):
	"""
	return error on weighted mean.
	"""
	
	if len(xerr)==1:
		return xerr
	
	else:
		model = np.ones(len(xerr))
		W = 1./(np.cov(xerr.T))
		return (1./np.dot(np.dot(model.T, W),model))**0.5

def w_mean(x, xerr):
	"""
	return weighted mean.
	"""
	if len(x)==1:
		return x
	else:
		return np.sum(x/xerr**2)/np.sum(1./xerr**2)

def w_std(xerr):
	"""
	return error on weighted mean.
	"""
	
	if len(xerr)==1:
		return xerr
	
	else:
		return (1./np.sum(1./xerr**2))**0.5
		#return np.mean(xerr)


def plot_style1(x, y, yerr, title, sim=0):

	if sim=='sim':
		# z_2pcf = np.load('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/2pcf/mice_z.npy')
		# b_2pcf = np.load('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/2pcf/mice_b.npy')
		# berr_2pcf = np.load('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/2pcf/mice_berr.npy')
		z_2pcf = np.loadtxt('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/2pcf/mice_sv_crocce.txt')[:,0]
		b_2pcf = np.loadtxt('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/2pcf/mice_sv_crocce.txt')[:,1]
		berr_2pcf = np.loadtxt('/Users/chihwaychang/Desktop/Work/bias/kappa_bias/2pcf/mice_sv_crocce.txt')[:,3]
		mplot.errorbar(z_2pcf, b_2pcf , yerr=berr_2pcf, fmt='o', color='grey', marker='s', alpha=0.5, label='2PCF')

	mplot.errorbar(x, y, yerr=yerr, fmt='o', label='mean', color='k', marker='o', markersize=10)
	mplot.title(title, fontsize=15)
	mplot.xlim(0.0, 1.2)
	mplot.ylim(0.2, 3.0)
	mplot.xlabel('z', fontsize=14)
	mplot.ylabel('b(z)', fontsize=14)
	#mplot.legend(loc='upper left', fontsize=14)
	mplot.xticks(fontsize = 12)
	mplot.yticks(fontsize = 12)

	return

def linear_func(x, b, c):
	return b*x+c
	
def fit_line(x, y, yerr):
	x0 = np.array([0.01, 1.])
	opt, cov = optimization.curve_fit(linear_func, x, y, x0, yerr, absolute_sigma=True)
	return opt, cov


def make_random_gal_catalog(mask,  maglim, ramin, ramax, decmin, decmax, N=100000000):
	"""
	return a random catalog of positions given the desired mask 
	and number of particles within the ra/dec boundaries.
	"""
	if mask != None:
		mask = hp.read_map(mask, nest=True)
		npix = mask.size
		nside = hp.npix2nside(npix)

	ra_rand = (np.random.random(N)* (ramax - ramin)) + ramin
	# random in cosine
	v = np.random.random(N)
	v *= 2
	v -= 1
	dec_rand = np.arccos(v)
	np.rad2deg(dec_rand, dec_rand)
	dec_rand -= 90.0
	dec_rand_sel = (dec_rand < decmax)*(dec_rand > decmin)
	ra_rand, dec_rand = ra_rand[dec_rand_sel], dec_rand[dec_rand_sel]

	if mask != None:
		# convert ra/dec into radians
		theta_rand = (90.0 - dec_rand)*np.pi/180.
		phi_rand = ra_rand*np.pi/180.
		pix_rand = hp.ang2pix(nside, theta_rand, phi_rand, nest=True)
		good_mask, = np.where(mask[pix_rand] > maglim)

		return ra_rand[good_mask], dec_rand[good_mask]

	if mask == None:
		return ra_rand, dec_rand


def plot_map_grid(h_dec, v_ra, ra_ref, decmin, decmax, color):
	# build the lines:
	h_lines_x = []
	h_lines_y = []
	v_lines_x = []
	v_lines_x_proj = []
	v_lines_y = []
	RA_label = []

	for i in range(len(h_dec)):
	    h_lines_x.append(np.arange(400)*0.1 - 20 + ra_ref)
	    h_lines_y.append(np.ones(400)*h_dec[i])
	    mplot.plot(np.arange(400)*0.2 - 20 + ra_ref, np.ones(400)*h_dec[i], color=color, lw=0.5)

	for i in range(len(v_ra)):
	    v_lines_x.append(np.ones(400)*v_ra[i])
	    v_lines_x_proj.append(np.ones(400)*(v_ra[i]-ra_ref)*np.cos((np.arange(400)*0.2 - 40 + (decmin+decmax )/2)/180*np.pi)+ ra_ref)
	    v_lines_y.append(np.arange(400)*0.2 - 40 + (decmin+decmax )/2)
	    mplot.plot(v_lines_x_proj[i], v_lines_y[i], color=color, lw=0.5)
	    RA_label.append(v_lines_x_proj[i][160])

	labels = [] 
	for i in range(len(v_ra)):
		labels.append(str(v_ra[i]))
	mplot.xticks(RA_label, labels, fontsize=14) 
	mplot.yticks(fontsize=14) 

