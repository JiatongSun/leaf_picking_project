import numpy as np
import cv2
import os

def getData():
    count = 0
    count2 = 0
    for filename in os.listdir("train/origin"):
        for mask_filename in os.listdir("train/mask"):
            if (filename in mask_filename) is False: continue
            count2 += 1
            if count2 <= 30 and count2 >= 15: continue
            print(f"load {filename}")
            count += 1
            image = cv2.imread(os.path.join("train/origin",filename))
            mask = cv2.imread(os.path.join("train/mask",mask_filename))
            nx,ny,ch = image.shape
            image = np.reshape(image,(nx*ny,ch))
            mask = np.reshape(mask,(nx*ny,ch))
            if count==1:
                stack = image[mask[:,0]!=0].copy()
            else:
                stack = np.vstack((stack,image[mask[:,0]!=0]))
    return np.array(stack)

def gaussian(data,mean,cov):
        cov_inv = np.linalg.inv(cov)
        diff = np.matrix(data-mean)
        N = (2.0 * np.pi) ** (-len(data[1]) / 2.0) * (1.0 / (np.linalg.det(cov) ** 0.5)) *\
            np.exp(-0.5 * np.sum(np.multiply(diff*cov_inv,diff),axis=1))
        return N

def GMM(data,K):
    
    n_feat = data.shape[0] 
    n_obs = data.shape[1]

    def initialize():
        mean = np.array([data[np.random.choice(n_feat,1)]],np.float64)
        cov = [np.random.randint(1,255)*np.eye(n_obs)]
        cov = np.matrix(np.multiply(cov,np.random.rand(n_obs,n_obs)))
        return {'mean': mean, 'cov': cov}
    
    bound = 0.0001
    max_itr = 500
    parameters = [initialize() for cluster in range (K)]
    cluster_prob = np.ndarray([n_feat,K],np.float64)
    
    #EM - step E
    itr = 0
    mix_c = [1./K]*K
    log_likelihoods = []
    while (itr < max_itr):
        print(itr)
        itr+=1
        for cluster in range (K):
            cluster_prob[:,cluster:cluster+1] = gaussian(data,parameters[cluster]['mean'],parameters[cluster]['cov'])*mix_c[cluster]
            
        cluster_sum = np.sum(cluster_prob,axis=1)
        log_likelihood = np.sum(np.log(cluster_sum))
        log_likelihoods.append(log_likelihood)
        cluster_prob = np.divide(cluster_prob,np.tile(cluster_sum,(K,1)).transpose())
        Nk = np.sum(cluster_prob,axis = 0) #2
        #EM - step M
        for cluster in range (K):
            new_mean = 1./ Nk[cluster]* np.sum(cluster_prob[:,cluster]*data.T,axis=1).T
            parameters[cluster]['mean'] = new_mean
            diff = data - parameters[cluster]['mean']
            new_cov = np.array(1./ Nk[cluster]*np.dot(np.multiply(diff.T,cluster_prob[:,cluster]),diff)) 
            parameters[cluster]['cov'] = new_cov
            mix_c[cluster] = 1./ n_feat * Nk[cluster]
            
       #log likelihood
        if len(log_likelihoods)<2: continue
        if np.abs(log_likelihood-log_likelihoods[-2])<bound : break

    return mix_c,parameters

if __name__ == '__main__':
    train_data = getData()
    K = 4
    mix_c,parameters = GMM(train_data,K)
    np.save('weights.npy',mix_c)
    np.save('parameters.npy',parameters)