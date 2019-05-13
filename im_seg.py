import numpy as np
import skimage
from sklearn.cluster import MeanShift, spectral_clustering, KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from operator import itemgetter
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import skimage.segmentation
import skimage.morphology
import skimage.filters
from sklearn.metrics import pairwise_distances
import scipy
from skimage.measure import label
from sklearn.manifold import spectral_embedding




# Mean Shift

def k(x,kernel='gaussian'):
    if(kernel == 'gaussian'):
        return 0.5*np.exp(-x)
    
    if(kernel == 'Epanechnikov'):
        return (np.abs(x) < 1) * (1 - x)



def g(x,kernel='gaussian'):
    if(kernel == 'gaussian'):
        return 0.5*np.exp(-x)

    if(kernel == 'Epanechnikov'):
        return (np.abs(x) < 1)*1.0    



def M(Xs,xs,hs,Xr,xr,hr,kernel='gaussian'):
    norm_s = (np.linalg.norm((xs.reshape(1,-1)-Xs),axis=1)/hs)**2
    norm_r = (np.linalg.norm((xr.reshape(1,-1)-Xr),axis=1)/hr)**2

    ks = k(norm_s,kernel)
    gs = g(norm_s,kernel)
    kr = k(norm_r,kernel)
    gr = g(norm_r,kernel)

    Ms = (np.einsum('ik,i,i->k',Xs,gs,kr)/np.sum(gs*kr)) - xs
    Mr = (np.einsum('ik,i,i->k',Xr,gr,ks)/np.sum(gr*ks)) - xr
    
    return Ms,Mr


def extract_basic_features(im): # 3 x H x W
    H,W,C = im.shape
        
    Xs = np.zeros((H*W,2))
    Xr = np.zeros((H*W,4))

    illu_feature = 0.5 + np.log(im[:,:,1] + 1e-20) - 0.5 * np.log(im[:,:,2] + 1e-20) - (1 - 0.5) * np.log(im[:,:,0] + 1e-20)

    im_luv = skimage.color.rgb2luv(im)
    Xr[:,0:3] = im_luv.reshape(-1,3)
    Xr[:,3] = illu_feature.reshape(-1)

    x = np.arange(0,W)
    x = np.tile(x,H)

    y = np.arange(0,H)
    y = np.repeat(y,W)

    Xs[:,0] = x
    Xs[:,1] = y


    return Xs, Xr


def mss(Xs,xs,hs,Xr,xr,hr,kernel='gaussian'):
    Ms = 1
    Mr = 1
    x1 = xs
    x2 = xr
    while(np.linalg.norm(Ms) > 0.001*hs or np.linalg.norm(Mr) > 0.001*hr):
        Ms,Mr = M(Xs,x1,hs,Xr,x2,hr,kernel)
        x1 = x1 + Ms
        x2 = x2 + Mr

    return x1,x2

def segmentation_mss(im,kernel='gaussian',hs=8,hr=5,Mval=60):

    Xs,Xr = extract_basic_features(im)
    out = []
    print (Xs.shape)
    print (Xr.shape)


    for i in range(len(Xs)):
        xs,xr = mss(Xs,Xs[i],hs,Xr,Xr[i],hr,kernel)
        out.append(np.concatenate((xs,xr)))
    out = np.array(out)
    print (out.shape)
    
    
    
    nbr_s = NearestNeighbors(radius=hs).fit(out[:,0:2])
    nbr_r = NearestNeighbors(radius=hr).fit(out[:,2:])
    
    unique = np.ones(len(out))
    
    for i in range(len(unique)):
        if(unique[i]):
            s_nbr = nbr_s.radius_neighbors([out[i,0:2]],return_distance=False)[0]
            r_nbr = nbr_r.radius_neighbors([out[i,2:]],return_distance=False)[0]
            nbr = list(set(s_nbr).intersection(r_nbr))
            unique[nbr] = 0
            unique[i] = 1
    
    print (np.sum(unique))
    cluster_centers = out[unique.astype('bool')]
    print (len(cluster_centers))
    
    
    
    nbr = NearestNeighbors(radius=Mval).fit(cluster_centers)
    
    unique = np.ones(len(cluster_centers))
    
    for i in range(len(unique)):
        if(unique[i]):
            nbrs = nbr.radius_neighbors([cluster_centers[i]],return_distance=False)[0]
            unique[nbrs] = 0
            unique[i] = 1
    
    
    
    print (np.sum(unique))
    cluster_centers = cluster_centers[unique.astype('bool')]
    print (len(cluster_centers))
    
    nbr = NearestNeighbors(n_neighbors=1).fit(cluster_centers)
    labels = nbr.kneighbors(out,return_distance=False) ## Cluster Indices
    
    labels = label(labels.reshape(H,W))

    return labels, cluster_centers




## Spectral - Normalized N-Cut

def w_gaussian(x,sigma,r):
    return (np.exp(-x**2)/sigma**2)*(x < r)


def ncut_merge(labels,W,a,b):
    ncut = 0
    if(a < b):
        labels += (labels==b)*(a-b)
    else:
        labels += (labels==a)*(b-a)

    return calculate_ncut(labels,W)



def find_split(labels,W,l):

    y = np.zeros(len(labels))

    # Split at 0
    #y = (labels>0)*1
    
    # Split at median
    m = np.median(labels)
    y = (labels>m)*1

    # Split at l
    #min_ncut = 1e6
    #y_temp = np.zeros(len(labels))
    #step = (np.max(labels) - np.min(labels))/l
    #i = np.min(labels)
    #while(i < np.max(labels)):      
    #    y_temp = (labels>i)*1
    #    ncut_temp = calculate_ncut(y_temp,W)
    #    if(ncut_temp < min_ncut):
    #        min_ncut = ncut_temp
    #        y = y_temp
    #    i = i + step
 
    return y



def calculate_ncut(labels,W):
    ncut = 0
    n_clusters = len(np.unique(labels))
    for i in range(n_clusters):
        cut_av = np.sum(W*(labels == i).reshape(-1,1)*(1-(labels==i)).reshape(1,-1))
        ass_a = np.sum(W*(labels == i).reshape(-1,1))
        ncut += cut_av/(ass_a+1e-10)
    return ncut


def find_labels(cluster_centers,n_pixels,W,l):
    y = np.zeros(n_pixels)
    for i in range(len(cluster_centers)):
        y_temp = find_split(cluster_centers[i],W,l)
        y += (y_temp * (y == 0))*(i+1)

    return y 


def exclude_unstable_eigenvectors(eig):
    maxv = np.amax(eig,axis=0)
    minv = np.amin(eig,axis=0)
    stable = (minv/maxv)
    stable = stable < 0.006
    return eig.T[stable.reshape(-1)].T

def spectral_seg(im,sigma=6,r=6,n_clusters=5):

    Height,Width,C = im.shape

    # Exract Features
    Xs,Xr = extract_basic_features(im)
    Xp = im.reshape(-1,3)
    im_hsv = skimage.color.rgb2hsv(im)
    Xq = im_hsv.reshape(-1,3)
 
    # Pairwise feature W k1*k2
    Ws = pairwise_distances(Xs)
    Wr = pairwise_distances(Xr)
    Wp = pairwise_distances(Xp)
    Wq = pairwise_distances(Xq)
    Wr = Wr*np.max(Ws)/np.max(Wr)
    Wp = Wp*np.max(Ws)/np.max(Wp)
    Wq = Wq*np.max(Ws)/np.max(Wq)

    Ws = w_gaussian(Ws,sigma=sigma,r=r) 
    Wr = w_gaussian(Wr,sigma=sigma,r=1e6) 
    Wp = w_gaussian(Wp,sigma=sigma,r=1e6) 
    Wq = w_gaussian(Wq,sigma=sigma,r=1e6) 
       
    #W = Ws*Wq*Wp
    W = Ws*Wq
 
    # Create D from W
    D = np.diag(np.sum(W,axis=1))


    # Solve generalized eigenvalue system
    e,vr = scipy.linalg.eigh(a=D-W, b=D)
    vr = exclude_unstable_eigenvectors(vr)
    print (vr.shape)

    if(n_clusters == 2):
        y = find_split(vr[:,1],W,5)

    else:

        # Cluster Eigenvalues
        embed = vr.T[0:100]
        clusters = KMeans(n_clusters = 10).fit(embed)
        cluster_centers = clusters.cluster_centers_



        ####### Merge Clusters ###########
        
        n_clusters_temp = len(cluster_centers)
        labels_temp = find_labels(cluster_centers,len(vr),W,10)

        for i in range(n_clusters_temp-n_clusters+1):
            index1 = -1
            index2 = -1
            min_ncut = 1e6 
            for j in range(len(cluster_centers)):
                for k in range(j+1,len(cluster_centers)):
                    ncut_temp = ncut_merge(labels_temp,W,j,k)
                    if(ncut_temp < min_ncut):
                        min_ncut = ncut_temp
                        index1 = j
                        index2 = k
            cluster_centers[index1] = (cluster_centers[index1]+cluster_centers[index2])/2
            labels_temp += (labels_temp==index2)*(index1-index2)
            cluster_centers = np.delete(cluster_centers,index2,0)


        # Compute segments
        y = find_labels(cluster_centers,len(vr),W,10)

 


    # Ncut
    ncut = calculate_ncut(y,W)

#    # Label the image
    labels = y.reshape(Height,Width)



    return labels, ncut
    



im = skimage.io.imread('../Data/Berkeley/BSDS300/images/train/12003.jpg')
print (im.shape)
im = skimage.transform.rescale(im, (1.0/9.0, 1/13.0), anti_aliasing=False,multichannel=True)
H,W,C = im.shape 


####################### Written MSS #####################
# Pros: 
#  Separate Bandwidth for spatial and color features
#  Number of clusters not required
#  h_s, h_r and M vary less between images than K

# Cons:
#  Not as good as GMM or Bayesian GMM with known K
#  Too Slow

#kernel = 'gaussian'
kernel = 'Epanechnikov'
hs = 8
hr = 5
Mval = 60
#clusters,cluster_centers = segmentation_mss(im,kernel=kernel,hs=hs,hr=hr,Mval=Mval)
#print (len(np.unique(clusters)))



####################### Function MSS ###################
#Cons: Bad performance on same bandwidth for spatial and color features

#Xs,Xr = extract_basic_features(im)
#clustering = MeanShift(bandwidth=12,min_bin_freq=20).fit(np.concatenate((Xs,Xr),axis=1))
#clustering = MeanShift(bandwidth=5,bin_seeding=True).fit(Xr)
#clusters = clustering.labels_.reshape(H,W)
#print (len(np.unique(clusters)))



################### GMM ################################
#Pros:
  # Good performance with known K
  # Fast
#Cons:
  # K changes for every image

#K = 5
#Xs,Xr = extract_basic_features(im)
#clusters = GaussianMixture(n_components=K).fit_predict(np.concatenate((Xs,Xr),axis=1)).reshape(H,W)
#print (len(np.unique(clusters)))



################### Bayesian DP prior Gaussian Mixture #######
# Pros:
#  Good performance wih known K
# Cons:
#  DP doesn't estimate K so well

#Xs,Xr = extract_basic_features(im)
#X = np.concatenate((Xs,Xr),axis=1)
#mean_prior = np.random.normal(loc=np.mean(X,axis=0),scale=100,size = X.shape[1])
#covariance_prior = 
#clusters= BayesianGaussianMixture(n_components=100, mean_prior = mean_prior, covariance_prior=None).fit_predict(np.concatenate((Xs,Xr),axis=1)).reshape(H,W)
#print (len(np.unique(clusters)))





## Spectral
clusters, ncut = spectral_seg(im,n_clusters=4)
print(ncut)
print (np.unique(clusters))



clusters = skimage.transform.rescale(clusters, (9.0,13.0), anti_aliasing=False,multichannel=False)
fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.imshow(clusters,alpha=1,cmap='hsv')
plt.show()
