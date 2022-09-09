import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import scipy.sparse as sp
import os
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std

def load_fake_data(path,prefix, pca_use, normalize=True):
    embedding_path = 'processed/'+prefix+'/'

    #load the polluted feats
    feats_polluted = sp.load_npz(embedding_path+'polluted_feats.npz')
    #load the polluted embeddings
    embeds_polluted = np.load(embedding_path+ 'dirty_embeddings.npy')
    '''
    embeds_polluted = sp.load_npz(embedding_path + 'dirty_embeddings_ext.npz')
    embeds_polluted = embeds_polluted.toarray()
    embeds_polluted = embeds_polluted[:,0:10]
    '''
    #print out the polluted embeddings
    #print(embeds_polluted)

    #change the csr_matrix to ndarray
    feats_polluted = feats_polluted.toarray()

    #concate the polluted features and embeddings
    #pca = PCA(n_components=50)
    #feats_polluted = pca.fit_transform(feats_polluted)
    feats = np.concatenate((feats_polluted,embeds_polluted), axis=1)
    #feats = feats_polluted

    #load the polluted_node_index
    polluted_idx = np.load(embedding_path+ 'polluted_node_index.npy')
    polluted_idx = polluted_idx.astype(int)

    feats = feats[polluted_idx,:]



    #For some datasets, the z_score method can benefit the training
    #need to be verified in the experiments
    #x_stats = {'mean': np.mean(feats), 'std': np.std(feats)}
    #feats = z_score(feats , x_stats['mean'], x_stats['std'])
    if pca_use:
        pca = PCA(n_components=100)
        feats = pca.fit_transform(feats)

    #norm = MinMaxScaler().fit(feats)
    #feats = norm.transform(feats)

    return feats
