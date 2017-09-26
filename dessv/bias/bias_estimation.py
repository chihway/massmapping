
import numpy as np 
import pylab as mplot
import sys
sys.path.append('../utils')
import kmeans_radec
from numpy.linalg import inv

def get_bias(map1, map2, mask):
	""" 
	Calculate zero-lag correlation between two maps. Here we calculate:
	<map2 map2> / <map1 map2>	
	"""
	map1[mask==1] = map1[mask==1] - np.mean(map1[mask==1])
	map2[mask==1] = map2[mask==1] - np.mean(map2[mask==1])
	bias = np.mean(map2[mask==1]*map2[mask==1])/np.mean(map1[mask==1]*map2[mask==1])
	return bias, np.mean(map2[mask==1]*map2[mask==1]), np.mean(map1[mask==1]*map2[mask==1])

def get_bias_denoise(map1, map2, map2_ran, mask):
    """ 
    Same as get_bias, but add noise-correction factors, so in the 
    end we calculate:
    (<map2 map2>-<map2_ran map2_ran>) / (<map1 map2>m)
    """

    # get random shot noise term from mean of N_ran randoms
    N_ran = len(map2_ran)
    map2_noise = 0.0

    for i in range(N_ran):
        map2_ran[i][mask==1] -= np.mean(map2_ran[i][mask==1])

        map2_noise += np.mean(map2_ran[i][mask==1]**2)
        # map1map2_noise += np.mean(map2_ran[i][mask==1]*map1[mask==1])
        # print(np.mean(map1_ran[i][mask==1]*map2[mask==1]), np.mean(map1_ran[i][mask==1]*map2[mask==1])/np.mean(map1[mask==1]*map2[mask==1]))

    map2_noise /= N_ran

    bias = (np.mean(map2[mask==1]*map2[mask==1])-map2_noise)/np.mean(map1[mask==1]*map2[mask==1])
    return 1./bias, (np.mean(map2[mask==1]*map2[mask==1])-map2_noise), np.mean(map1[mask==1]*map2[mask==1])

def get_jk_simple(map1, map1_ran, map2, map2_ran, mask, area, bias_model, pix, frac=0.5):
    """
    Simple gridding for jackknife samples.
    """

    bin_size = int(area**0.5*60./pix)
    nsidex = (len(mask[0])/bin_size)+1
    nsidey = (len(mask)/bin_size)+1
    print(bin_size, nsidex, nsidey)

    jk_sample = []
    for i in range(nsidey):
    	for j in range(nsidex):
            mask_temp = mask*1.0
            if (i!=(nsidey-1)) and (j!=(nsidex-1)):
            	mask_temp[i*bin_size:(i+1)*bin_size, j*bin_size:(j+1)*bin_size]=0
            if (i!=(nsidey-1)) and (j==(nsidex-1)):
            	mask_temp[i*bin_size:(i+1)*bin_size, j*bin_size:]=0
            if (i==(nsidey-1)) and (j!=(nsidex-1)):
            	mask_temp[i*bin_size:, j*bin_size:(j+1)*bin_size]=0
            if (i==(nsidey-1)) and (j==(nsidex-1)):
            	mask_temp[i*bin_size:, j*bin_size:]=0
            if (np.sum(mask - mask_temp)>=(frac*bin_size*bin_size)):
                if bias_model == 'bias':
                    jk_sample.append(get_bias(map1, map2, mask_temp)[0])
                if bias_model == 'bias_denoise':
                    jk_sample.append(get_bias_denoise(map1, map1_ran, map2, map2_ran, mask_temp)[0])
                if bias_model == 'cross':
                    jk_sample.append(crosscorr(map1, mask_temp))
    return jk_sample


def get_jk_kmean(map1, map2, map2_ran, mask, jk_label, bias_model, nn):
    """
    Kmeans gridding for jackknife samples.
    """

    jk_label = np.load(jk_label)['labels']
    N_jk = int(np.max(jk_label)+1)
    jk_sample = []
    for i in range(N_jk):
        # print('jk',i)
        mask_temp = mask*1.0
        mask_temp[jk_label==i] = 0
        if bias_model == 'bias':
            jk_sample.append(get_bias(map1, map2, mask_temp)[0])
        if bias_model == 'bias_denoise':
            jk_sample.append(get_bias_denoise(map1, map2, map2_ran, mask_temp)[nn])
        if bias_model == 'cross':
            jk_sample.append(crosscorr(map1, mask_temp))
        if bias_model == 'test':
            jk_sample.append(variance(map1, mask_temp))
        if bias_model == 'm':
            jk_sample.append(getM(map1, map2, map2_ran, mask_temp))
    return jk_sample


def jackknife_bias(map1, map2, map2_ran, mask, area, pix, bias_model, jk_label):
    """ 
    Calculate zero-lag correlation between two maps, with 
    jackknife errors. Jackknife samples can be constructed in a few ways

    return: bias, <kgkg>, <k kg>, mean_bias_jk, std_bias_jk, n_jk
    """

    if bias_model == 'bias':
        bias = get_bias(map1, map2, mask)
    if bias_model == 'bias_denoise':
        bias = get_bias_denoise(map1, map2, map2_ran, mask)

    #bias_jk = get_jk_simple(map1, map2, map2_ran, mask, area, bias_model, pix, frac=0.5)
    bias_jk = get_jk_kmean(map1, map2, map2_ran, mask, jk_label, bias_model, 0)
    return bias[0], bias[1], bias[2], np.mean(bias_jk), (len(bias_jk)-1)**0.5*np.std(bias_jk), len(bias_jk)

def crosscorr(map, mask):
    """
    Calculate one cross-correlation matrix.
    """
    cross = np.zeros((len(map), len(map[0])))

    for j in range(len(map)):
        for i in range(len(map[0])):
            if j>=i:
                A = 0
                for k in range(len(map[0])):
                    A += np.mean(map[j][i][mask==1]*map[j][k][mask==1])
                cross[j][i] = (A/np.mean(map[j][i][mask==1]*map[j][i][mask==1]))
    return cross


def jackknife_variance(map1, mask, jk_label):
    """ 
    Calculate zero-lag variance of a map, with jackknife errors. 
    """

    var = variance(map1, mask)
    var_jk = get_jk_kmean(map1, map1, map1, map1, mask, jk_label, 'test', 0)

    return var, np.mean(var_jk),  (len(var_jk)-1)**0.5*np.std(var_jk), len(var_jk)


def variance(map, mask):
    """
    Calculate variance of masked maps.
    """
    return np.mean(map[mask==1]**2)

def jackknife_crosscorr(map1, mask, area, pix, jk_label):
    """ 
    Calculate zero-lag cross-correlation between two kg maps, with 
    jackknife errors. 
    """

    cross = crosscorr(map1, mask)
    cross_jk = get_jk_kmean(map1, map1, map1, mask, jk_label, 'cross')

    return cross, np.mean(cross_jk, axis=0), (len(cross_jk)-1)**0.5*np.std(cross_jk, axis=0), len(cross_jk)

def getM(map, map_ran, M, mask):
    N = len(map)
    m = np.zeros((N,N))    
    measured = np.zeros((N,N))  
    recover = np.zeros((N,N))     

    for ii in range(N):
        for i in range(ii+1):
            measured[ii][i] =(np.mean(map[ii][i][mask==1]**2)-np.mean(map_ran[0][ii][i][mask==1]**2)) 
    for ii in range(N): 
        recover[ii][:ii+1] = np.dot(np.linalg.inv(M[ii*N:ii*N+ii+1,ii*N:ii*N+ii+1]), measured[ii][:ii+1])
    #print(measured)
    return (recover/measured)**0.5

def jackknife_m(map1, map1_ran, M, mask, area, pix, jk_label):
    """ 
    Calculate zero-lag cross-correlation between two kg maps, with 
    jackknife errors. 
    """

    m = getM(map1, map1_ran, M, mask)
    m_jk = get_jk_kmean(map1, map1_ran, M, mask, jk_label, 'm')

    return m, np.mean(m_jk, axis=0), (len(m_jk)-1)**0.5*np.std(m_jk, axis=0), len(m_jk)


def make_jk_id(N2d, edges, mask, ra_ref, ra_range, dec_range, ncen, maxiter, tol):
    """
    return grid of ids for jk sample.
    """
    radec = np.zeros((len(N2d.flatten()),2))
    Ngal = np.zeros((len(N2d.flatten()),1))
    ii = []
    jj = []
    nn = []
    for i in range(len(N2d)):
        for j in range(len(N2d[0])):
            ra_fixed = ((edges[1][j]+edges[1][j+1])/2 - ra_ref)/np.cos((edges[0][i]+edges[0][i+1])/2/180*np.pi)+ra_ref
            dec_fixed = (edges[0][i]+edges[0][i+1])/2
            Ngal[i*len(N2d[0])+j][0] = mask[i][j]
            radec[i*len(N2d[0])+j][0] = ra_fixed
            radec[i*len(N2d[0])+j][1] = dec_fixed
            ii.append(i)
            jj.append(j)

    print('Apply mask...')
    w, = np.where(  (radec[:,0] > ra_range[0])
                    & (radec[:,0] < ra_range[1])
                    & (radec[:,1] > dec_range[0])
                    & (radec[:,1] < dec_range[1])
                    &  (Ngal[:,0] > 0) )

    # now run kmeans
    km = kmeans_radec.kmeans_sample(radec[w,:],
                                   ncen,
                                   maxiter=maxiter,
                                   tol=tol)
    if not km.converged:
         raise RuntimeError("k means did not converge")

    labels_grid_1d = np.zeros(len(ii))
    labels_grid_1d[w] = km.labels
    labels_grid_2d = N2d*0.0
    for i in range(len(w)):
        labels_grid_2d[ii[w[i]]][jj[w[i]]] = labels_grid_1d[w[i]]

    return labels_grid_2d

def covariance_bias(nparam, njk, Nbin_z_l, Nbin_z_s, map1, map2, map2_ran, map3, map4, map4_ran, mask, jk_file, bias_type, factor):
    
    """
    get bias and covariance
    """

    Bias = np.zeros(Nbin_z_l)
    Bias_err = np.zeros(Nbin_z_l) 
    
    Data = []
    Err = []

    for i in range(Nbin_z_l):

        N_m = nparam*(Nbin_z_s - i)
        model = np.ones(N_m)*1.0
        jk = np.zeros((N_m, njk))
        data = np.zeros(N_m)
        
        for j in range(Nbin_z_s):
            if j>=i:
                #print('lens bin: '+str(i)+'; source bin: '+str(j))
                data[j-i] = get_bias_denoise(map1[j], map2[j][i], map2_ran[:,j,i], mask)[0]/factor[j][i]
                jk[j-i] = np.array(get_jk_kmean(map1[j], map2[j][i], map2_ran[:,j,i], mask, jk_file, bias_type, 0))/factor[j][i]
                if nparam==2:
                    data[j-i+(Nbin_z_s - i)] = get_bias_denoise(map3[j], map4[j][i], map4_ran[:,j,i], mask)[0]/factor[j][i]
                    jk[j-i+(Nbin_z_s - i)] = np.array(get_jk_kmean(map3[j], map4[j][i], map4_ran[:,j,i], mask, jk_file, bias_type, 0))/factor[j][i]

        C = (np.cov(jk)*(njk-1))
        # if N_m>1:
        #     import scipy.linalg
        #     XX = scipy.linalg.eigvals(C)
        #     print(np.max(XX)/np.min(XX))
        
        if nparam==1 and i==Nbin_z_l-1:
            Err.append(C**0.5)
            Data.append(data)
            W = 1.0/C*(njk-N_m-2)/(njk-1)
     
        else:
            Err.append(np.diag(C)**0.5)
            Data.append(data)
            W = inv(C)*(njk-N_m-2)/(njk-1)  

        A = 1./np.dot(np.dot(model.T, W), model)*np.dot(np.dot(model.T, W), data)
        Aerr = (1./np.dot(np.dot(model.T, W),model))**0.5
        
        Bias[i] = A
        Bias_err[i] = Aerr
        
    # this is all inverse bias!!
    return Bias, Bias_err, Data, Err


# def covariance_bias_temp(nparam, njk, Nbin_z_l, Nbin_z_s, map1, map1_ran, map2, map2_ran, map3, map3_ran, map4, map4_ran, mask, jk_file, bias_type, factor):
    
#     """
#     get bias and covariance
#     """

#     Bias1 = np.zeros(Nbin_z_l)
#     Bias_err1 = np.zeros(Nbin_z_l) 
#     Data1 = []
#     Err1 = []

#     Bias2 = np.zeros(Nbin_z_l)
#     Bias_err2 = np.zeros(Nbin_z_l) 
#     Data2 = []
#     Err2 = []

#     for i in range(Nbin_z_l):

#         N_m = nparam*(Nbin_z_s - i)
#         model = np.ones(N_m)*1.0
#         jk1 = np.zeros((N_m, njk))
#         data1 = np.zeros(N_m)
#         jk2 = np.zeros((N_m, njk))
#         data2 = np.zeros(N_m)

#         for j in range(Nbin_z_s):
#             if j>=i:
#                 #print('lens bin: '+str(i)+'; source bin: '+str(j))
#                 data1[j-i] = get_bias_denoise(map1[j], map1_ran[:,j], map2[j][i], map2_ran[:,j,i], mask)[1]*factor[j][i]
#                 jk1[j-i] = np.array(get_jk_kmean(map1[j], map1_ran[:,j], map2[j][i], map2_ran[:,j,i], mask, jk_file, bias_type, 1))*factor[j][i]
#                 data2[j-i] = get_bias_denoise(map1[j], map1_ran[:,j], map2[j][i], map2_ran[:,j,i], mask)[2]
#                 jk2[j-i] = np.array(get_jk_kmean(map1[j], map1_ran[:,j], map2[j][i], map2_ran[:,j,i], mask, jk_file, bias_type, 2))

#                 if nparam==2:
#                     data1[j-i+(Nbin_z_s - i)] = get_bias_denoise(map3[j], map3_ran[:,j], map4[j][i], map4_ran[:,j,i], mask)[1]*factor[j][i]
#                     jk1[j-i+(Nbin_z_s - i)] = np.array(get_jk_kmean(map3[j], map3_ran[:,j], map4[j][i], map4_ran[:,j,i], mask, jk_file, bias_type, 1))*factor[j][i]
#                     data2[j-i+(Nbin_z_s - i)] = get_bias_denoise(map3[j], map3_ran[:,j], map4[j][i], map4_ran[:,j,i], mask)[2]
#                     jk2[j-i+(Nbin_z_s - i)] = np.array(get_jk_kmean(map3[j], map3_ran[:,j], map4[j][i], map4_ran[:,j,i], mask, jk_file, bias_type, 2))


#         C = (np.cov(jk1)*(njk-1))
#         if nparam==1 and i==Nbin_z_l-1:
#             Err1.append(C**0.5)
#             Data1.append(data1)
#             W = 1.0/C*(njk-N_m-2)/(njk-1)       
#         else:
#             Err1.append(np.diag(C)**0.5)
#             Data1.append(data1)
#             W = inv(C)*(njk-N_m-2)/(njk-1)  
        
#         A = 1./np.dot(np.dot(model.T, W), model)*np.dot(np.dot(model.T, W), data1)
#         Aerr = (1./np.dot(np.dot(model.T, W),model))**0.5    
#         Bias1[i] = A
#         Bias_err1[i] = Aerr

#         C = (np.cov(jk2)*(njk-1))
#         if nparam==1 and i==Nbin_z_l-1:
#             Err2.append(C**0.5)
#             Data2.append(data2)
#             W = 1.0/C*(njk-N_m-2)/(njk-1)       
#         else:
#             Err2.append(np.diag(C)**0.5)
#             Data2.append(data2)
#             W = inv(C)*(njk-N_m-2)/(njk-1)  
        
#         A = 1./np.dot(np.dot(model.T, W), model)*np.dot(np.dot(model.T, W), data2)
#         Aerr = (1./np.dot(np.dot(model.T, W),model))**0.5    
#         Bias2[i] = A
#         Bias_err2[i] = Aerr

#     Bias = Bias1/Bias2
#     Bias_err = (Bias_err1**2/Bias2 + Bias1*(Bias_err2/Bias2**2)**2)**0.5
    
#     Data1 = np.array(Data1)
#     Data2 = np.array(Data2)
#     Err1 = np.array(Err1)
#     Err2 = np.array(Err2)
#     Data = Data1/Data2
#     Err = (Err1**2/Data2 + Data1*(Err2/Data2**2)**2)**0.5

#     return Bias, Bias_err, Data, Err


def zerolag_bias(nparam, Nbin_z_l, Nbin_z_s, map1, map2, map2_ran, map3, map4, map4_ran, mask, bias_type, factor):

    """
    get bias
    """

    Data = []
    M1 = []
    M2 = []

    for i in range(Nbin_z_l):

        N_m = nparam*(Nbin_z_s - i)
        model = np.ones(N_m)*1.0
        data = np.zeros(N_m)
        m1 = np.zeros(N_m)
        m2 = np.zeros(N_m)

        for j in range(Nbin_z_s):
            if j>=i:
                #print('lens bin: '+str(i)+'; source bin: '+str(j))
                temp = get_bias_denoise(map1[j], map2[j][i], map2_ran[:,j,i], mask)
                data[j-i] = temp[0]/factor[j][i]
                m1[j-i] = temp[1]/factor[j][i]
                m2[j-i] = temp[2]

                if nparam==2:
                    temp = get_bias_denoise(map3[j], map4[j][i], map4_ran[:,j,i], mask)
                    data[j-i+(Nbin_z_s - i)] = temp[0]/factor[j][i]
                    m1[j-i+(Nbin_z_s - i)] = temp[1]/factor[j][i]
                    m2[j-i+(Nbin_z_s - i)] = temp[2]

        Data.append(data)
        M1.append(m1)
        M2.append(m2)

    return Data, M1, M2



