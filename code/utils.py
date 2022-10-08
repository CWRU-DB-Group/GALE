import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse.linalg import inv
import sys
import json
import os
from networkx.readwrite import json_graph as jg
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import faiss
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import timeit


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index




def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.toarray()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict




def dumpJSON(destDirect, datasetName, graph, idMap, classMap, features):
    print("Dumping into JSON files...")
    # Turn graph into data
    dataG = jg.node_link_data(graph)
    # print(graph.number_of_edges())
    # Make names
    json_G_name = destDirect + '/' + datasetName + '-G.json'
    json_ID_name = destDirect + '/' + datasetName + '-id_map.json'
    json_C_name = destDirect + '/' + datasetName + '-class_map.json'
    npy_F_name = destDirect + '/' + datasetName + '-feats'

    # Dump graph into json file
    with open(json_G_name, 'w') as outputFile:
        json.dump(dataG, outputFile)

    # Dump idMap into json file
    with open(json_ID_name, 'w') as outputFile:
        json.dump(idMap, outputFile)

    # Dump classMap into json file
    with open(json_C_name, 'w') as outputFile:
        json.dump(classMap, outputFile)

    # Save features as .npy file
    print("Saving features as numpy file...")
    np.save(npy_F_name, features)

    print("all part finished")


def entropy_score(prob_dist):
    ''' Entropy-based uncertainty sampling
    returns the uncertainty score of a probability distribution using entropy score
    keyword arguments:
    prob_list: a numpy array of real numbers between 0 and 1 that total to 1.0'''
    log_probs = prob_dist* np.log2(prob_dist, out=np.zeros_like(prob_dist), where=(prob_dist!=0))
    raw_entropy = 0 -np.sum(log_probs,axis=1)
    normalized_entropy = raw_entropy/ np.log2(prob_dist.shape[1])

    return normalized_entropy




def random_sampler(sample_size, train_active_mask, seed,ground_truth):
    print("start implementing random sampler")
    idx_range = np.where(train_active_mask==True)[0]
    initial_true = idx_range.shape[0]
    print("seed is {}".format(seed))
    np.random.seed(seed)
    already_selected = np.random.choice(idx_range, size=sample_size,replace=False)
    train_active_mask[already_selected]= False
    intersect = list(set(already_selected) & set(ground_truth))
    print("length of intersection with ground truth error is {}".format(len(intersect)))
    print("selected node id is {}".format(already_selected))
    assert (train_active_mask[train_active_mask==True].shape[0] == initial_true-sample_size)
    return already_selected, train_active_mask

def entropy_sampler(sample_size, train_active_mask, active_entropy_score,ground_truth):
    print("start implementing entropy sampler")
    idx_range = np.where(train_active_mask == True)[0]
    initial_true = idx_range.shape[0]
    #rank the active_entropy_score and only select top sample_size
    #indices = np.argsort(active_entropy_score)[::-1]
    indices = np.argsort(active_entropy_score)
    #print(active_entropy_score[indices[0]])
    #print(active_entropy_score[indices[1]])
    already_selected = idx_range[indices[:sample_size]]
    #print(already_selected)
    train_active_mask[already_selected] = False
    intersect = list(set(already_selected) & set(ground_truth))
    print("length of intersection with ground truth error is {}".format(len(intersect)))
    assert (train_active_mask[train_active_mask == True].shape[0] == initial_true - sample_size)
    return already_selected, train_active_mask

def margin_sampler(sample_size, train_active_mask, dis_active_data):
    print("start implementing entropy sampler")
    sort_distances = np.sort(dis_active_data, 1)[:, -2:]
    min_margin = sort_distances[:, 1] - sort_distances[:, 0]
    rank_ind = np.argsort(min_margin)
    idx_range = np.where(train_active_mask == True)[0]
    initial_true = idx_range.shape[0]

    already_selected = idx_range[rank_ind[:sample_size]]
    #print(already_selected)
    train_active_mask[already_selected] = False
    assert (train_active_mask[train_active_mask == True].shape[0] == initial_true - sample_size)
    return already_selected, train_active_mask


def kmeans_sampler(cluster_size, sample_size, train_active_mask, data_embeds,ground_truth):
    print("start implementing k-means sampler")
    start = timeit.default_timer()
    km = KMeans(
        n_clusters=cluster_size, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    y_km = km.fit_predict(data_embeds)
    centroids_1 = km.cluster_centers_
    #squared_dis = km.inertia_
    x_dist = km.transform(data_embeds)

    '''
    ##compute the nearest neighbor to centroids_1
    x_dist = km.transform(data_embeds)
    squared_dist = x_dist.sum(axis=1).round(2)
    ##from each cluster, compute the nearest neighbors with the centroid
    print(km.labels_)
    ##verify one instance squared_dist
    print(centroids_1[km.labels_[1]])
    print(data_embeds[1])
    temp1 = data_embeds[0]
    temp2 = centroids_1[km.labels_[0]]
    temp_dist = euclidean_distances(temp1.reshape(1,-1), temp2.reshape(1,-1))
    temp_verify = km.transform(data_embeds)[0][km.labels_[0]]
    print(temp_dist - temp_verify)
    ##check each row the smallest whether corresponds to the class label
    predicted_cluster=np.argmin(x_dist,axis=1)
    assigned_clusters = km.labels_
    assert np.array_equiv(predicted_cluster,assigned_clusters) ==True
    '''
    idx_range = np.where(train_active_mask == True)[0]
    initial_true = idx_range.shape[0]
    #initialize a dict to store the closest node
    # (1) distance to its centroid
    # (2) node id
    # within each cluster


    epoch_size = sample_size / cluster_size
    already_selected = np.zeros(shape=(sample_size,), dtype=int)
    already_selected_idx =[]
    count =0


    for epoch in range(sample_size):
        cluster_dist = dict()
        maxX = 1000000.0
        for i in range(cluster_size):
            cluster_dist[i] = [maxX, None]
        for i in range(y_km.shape[0]):
            pred_clus = y_km[i]
            #print("instance {0:} is assigned to {1:} cluster".format(idx_range[i],pred_clus))
            #print("euclidean distance is {:5.4f}".format(x_dist[i][pred_clus]))
            dist = x_dist[i][pred_clus]
            if cluster_dist[pred_clus][0] > dist:
                dist_arr = x_dist[i]
                #cluster_dist[pred_clus][0] = np.sort(dist_arr)[epoch]
                cluster_dist[pred_clus][0] = dist
                #print(np.argsort(dist_arr)[11]==y_km[i])
                cluster_dist[pred_clus][1] = idx_range[i]
        for i in range(len(cluster_dist.keys())):
            key = cluster_dist.keys()[i]
            dist_id = cluster_dist[key][1]
            ##skip some cluster if the remaining nodes don't fall into that cluster
            if dist_id!=None:
                already_selected[count] = cluster_dist[key][1]
                count+=1
                #print(count)
                already_selected_idx.append(cluster_dist[key][1])
            if count == sample_size:
                break

        ##update the y_km and idx_range to the remaining items
        removed_idx =[]
        for i in range(idx_range.shape[0]):
            if idx_range[i] in already_selected_idx:
                removed_idx.append(i)
        y_km = np.delete(y_km, removed_idx)
        idx_range = np.delete(idx_range, removed_idx)
        if count ==sample_size:
            break

    intersect = list(set(already_selected) &set(ground_truth))
    print("length of intersection with ground truth error is {}".format(len(intersect )))
    print("selected node id is {}".format(already_selected))
    train_active_mask[already_selected] = False
    assert (train_active_mask[train_active_mask == True].shape[0] == initial_true - sample_size)
    stop = timeit.default_timer()
    print("Time:", stop - start)
    return already_selected, train_active_mask

def approximation_sampler(cluster_size, sample_size, train_active_mask, data_embeds, ground_truth, lam):
    #lam = 0.0
    #lam= 0.0001
    print("start implementing approximation sampler")
    start = timeit.default_timer()
    print("calculating k-means first")
    km = KMeans(
        n_clusters=cluster_size, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    y_km = km.fit_predict(data_embeds)
    centroids = km.cluster_centers_
    x_dist = km.transform(data_embeds)


    #print("initialize the empty set S")
    #sample_set = set()
    sample_set = []
    typicality_dict = dict()
    train_active_idx = np.where(train_active_mask == True)[0]
    initial_true =  train_active_idx.shape[0]

    ##get a cluster_flage
    cluster_flag_dict = dict()


    for id, val in enumerate(set(train_active_idx.flatten())):
        ###get the corresponding euclidean dist to the centroid
        pred_cluster_val = y_km[id]
        cluster_flag_dict[pred_cluster_val] = False
        dist_val = x_dist[id][pred_cluster_val]
        #print("instance {0:} is assigned to {1:} cluster".format(val, pred_cluster_val))
        #print("euclidean distance is {:5.4f}".format(dist_val))
        if dist_val !=0.0:
            typicality_dict[val] = 1.0/dist_val
        else:
            typicality_dict[val] = 1.0/(dist_val+1e-4)


    ##pre-compute the embedding distance between any two node-pair
    euclid_dict = dict()

    '''
    pre_computing_begin = timeit.default_timer()
    for i in range(len(train_active_idx)):
        for j in range(len(train_active_idx)):
            index_node_source = i
            index_node_dest = j
            #print(index_node_source)
            #print(index_node_dest)
            source_embeds = data_embeds[index_node_source]
            dest_embeds = data_embeds[index_node_dest]
            source_embeds = np.reshape(source_embeds, (1,-1))
            dest_embeds = np.reshape(dest_embeds, (1,-1))
            euclid_dict[(index_node_source, index_node_dest)] =euclidean_distances(source_embeds, dest_embeds)
    pre_computing_end = timeit.default_timer()
    print("Precomputing Time :", pre_computing_end -pre_computing_begin)
    '''



    #dynamic_tain_active_idx = train_active_idx.copy()
    #dynamic_data_embeds = data_embeds.copy()


    non_negative_set_dict = dict()
    non_negative_set_dict[0]= 0.0
    ##initialize a n by len(sample_size) matrix to save dist computation
    dist_pair_array = np.zeros(shape=(len(train_active_idx), sample_size+1))

    while len(sample_set)< sample_size:
        #get all the node index from train_active_mask\sample_set
        #train_active_idx = np.where(train_active_mask==True)[0]
        remaining_idx_set = set(train_active_idx.flatten()).difference(set(sample_set))
        # prepare a 2-dimensional array to help select maximizing ojective
        objective_selected_dist = np.zeros(shape=(len(remaining_idx_set), 7))
        #find u \in train_active_mask\S maximizing phi'_u(S)
        start_epoch = timeit.default_timer()
        if len(sample_set )> 0:
            latest_added = sample_set[-1]
        for id , val in enumerate(remaining_idx_set):
            start_search = timeit.default_timer()
            #temp_set = sample_set.copy()
            temp_set = list(sample_set)
            #print("#######################")
            #print(id, val)
            ##y_km needs to be dynamically trimed
            pred_cluster_val = y_km[id]

            if cluster_flag_dict[pred_cluster_val]==False:
                #temp_set.add(val)
                temp_set.append(val)
                unprocess_first_term = non_negative_set(val, temp_set, typicality_dict, non_negative_set_dict)
                selected_sample_size = len(sample_set)
                if selected_sample_size not in non_negative_set_dict:
                    raise ValueError("The non_negative_set_dict doesn't save previous sample_size as key value!")
                non_negative_set_ind = unprocess_first_term - non_negative_set_dict[selected_sample_size]
                stop_search1 = timeit.default_timer()
                #print("Search Time 1:", stop_search1 - start_search)
                first_term = 0.5* non_negative_set_ind
                if len(sample_set )==0:
                    dist_ind = 0.0
                else:
                    dist_ind,dist_pair_array = optimized_dist(val, latest_added, sample_set, data_embeds,train_active_idx,euclid_dict, dist_pair_array)
                #dist_ind = dist(val, sample_set,data_embeds,train_active_idx,euclid_dict)
                objective = first_term + dist_ind*lam
                objective_selected_dist[id][0] = objective
                objective_selected_dist[id][1] = val
                objective_selected_dist[id][2] =pred_cluster_val
                objective_selected_dist[id][3] = id
                objective_selected_dist[id][4] = unprocess_first_term
                objective_selected_dist[id][5] = first_term
                objective_selected_dist[id][6] = dist_ind

                stop_search2 = timeit.default_timer()
                #print("Search Time 2:", stop_search2-start_search)
            else:
                continue
        ##sort the objective_selected_dist
        print("current sample size {:}".format(len(sample_set)))
        stop_epoch =timeit.default_timer()
        #print("Time:", stop_epoch-start_epoch)
        '''
        a =np.array([[9,2,3],
                  [4,5,5],[7,0,5]])
        b = [5]
        for ele in b:
            a =a[a[:,2]!=5,:]
        print a
        print(a[a[:,0].argsort()[::-1]])
        '''
        '''
        ##exclude the existing class
        for pred_c in range(np.max(y_km)+1):
            if cluster_flag_dict[pred_c]==True:
                objective_selected_dist =objective_selected_dist[objective_selected_dist[:,2]!=pred_c,:]
        '''
        objective_selected_dist = objective_selected_dist[objective_selected_dist[:,0].argsort()[::-1]]
        selected_idx = int(objective_selected_dist[0][1])
        #print("objective is {:.7f}".format(objective_selected_dist[0][0]))
        print("first_term is {:.7f}".format(objective_selected_dist[0][5]))
        print("second_term is {:.7f}".format(objective_selected_dist[0][6]))
        pred_class = int(objective_selected_dist[0][2])
        #sample_set.add(selected_idx)
        sample_set.append(selected_idx)
        ##update the non_negative_set_dict
        if len(sample_set) not in non_negative_set_dict:
            selected_sample_size = len(sample_set)
            non_negative_set_dict [selected_sample_size] = objective_selected_dist[0][4]
            ##update non_negative_set_dict dictionary for this updated sample size
            #non_negative_set_dict = update_non_negative_set_dict(selected_idx, sample_set, data_embeds, train_active_idx, euclid_dict, non_negative_set_dict)

        #print("add {} to sample_set".format(selected_idx))
        ##update the cluster_flag_dict
        if pred_class!=-1:
            cluster_flag_dict[pred_class] =True
        trimmed_id = int(objective_selected_dist[0][3])
        y_km = np.delete(y_km, trimmed_id, axis=0)
        ##reintialize cluster_flag_dict if all values ==True
        if all(value == True for value in cluster_flag_dict.values()) == True:
            for key in cluster_flag_dict.keys():
                if key not in set(y_km):
                    cluster_flag_dict.pop(key)
            cluster_flag_dict.update(dict(zip(cluster_flag_dict ,list(set(y_km)))))
            cluster_flag_dict = dict.fromkeys(cluster_flag_dict, False)

        #print(cluster_flag_dict.keys())

        #dynamic_data_embeds= np.delete(dynamic_data_embeds, np.where(dynamic_tain_active_idx  == selected_idx), axis=0)
        #dynamic_tain_active_idx =np.delete(dynamic_tain_active_idx,np.where(dynamic_tain_active_idx ==selected_idx))
    #already_selected = list(sample_set)
    already_selected = sample_set
    print("selected node id is {}".format(already_selected))
    train_active_mask[already_selected] = False
    assert (train_active_mask[train_active_mask == True].shape[0] == initial_true - sample_size)
    stop = timeit.default_timer()
    print("Time:", stop - start)
    return already_selected, train_active_mask


def approximation_memo_sampler(cluster_size, sample_size, train_active_mask, data_embeds, ground_truth, lam):
    #lam = 0.0
    #lam= 0.0001
    print("start implementing approximation memorization sampler")
    start = timeit.default_timer()
    print("calculating k-means first")
    km = KMeans(
        n_clusters=cluster_size, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    y_km = km.fit_predict(data_embeds)
    centroids = km.cluster_centers_
    x_dist = km.transform(data_embeds)


    #print("initialize the empty set S")
    #sample_set = set()
    sample_set = []
    typicality_dict = dict()
    train_active_idx = np.where(train_active_mask == True)[0]
    initial_true =  train_active_idx.shape[0]

    ##get a cluster_flage
    cluster_flag_dict = dict()


    for id, val in enumerate(set(train_active_idx.flatten())):
        ###get the corresponding euclidean dist to the centroid
        pred_cluster_val = y_km[id]
        cluster_flag_dict[pred_cluster_val] = False
        dist_val = x_dist[id][pred_cluster_val]
        #print("instance {0:} is assigned to {1:} cluster".format(val, pred_cluster_val))
        #print("euclidean distance is {:5.4f}".format(dist_val))
        if dist_val !=0.0:
            typicality_dict[val] = 1.0/dist_val
        else:
            typicality_dict[val] = 1.0/(dist_val+1e-4)


    ##pre-compute the embedding distance between any two node-pair
    euclid_dict = dict()

    '''
    pre_computing_begin = timeit.default_timer()
    for i in range(len(train_active_idx)):
        for j in range(len(train_active_idx)):
            index_node_source = i
            index_node_dest = j
            #print(index_node_source)
            #print(index_node_dest)
            source_embeds = data_embeds[index_node_source]
            dest_embeds = data_embeds[index_node_dest]
            source_embeds = np.reshape(source_embeds, (1,-1))
            dest_embeds = np.reshape(dest_embeds, (1,-1))
            euclid_dict[(index_node_source, index_node_dest)] =euclidean_distances(source_embeds, dest_embeds)
    pre_computing_end = timeit.default_timer()
    print("Precomputing Time :", pre_computing_end -pre_computing_begin)
    '''



    #dynamic_tain_active_idx = train_active_idx.copy()
    #dynamic_data_embeds = data_embeds.copy()


    non_negative_set_dict = dict()
    non_negative_set_dict[0]= 0.0
    ##initialize a n by len(sample_size) matrix to save dist computation
    dist_pair_array = np.zeros(shape=(len(train_active_idx), sample_size+1))

    while len(sample_set)< sample_size:
        #get all the node index from train_active_mask\sample_set
        #train_active_idx = np.where(train_active_mask==True)[0]
        remaining_idx_set = set(train_active_idx.flatten()).difference(set(sample_set))
        # prepare a 2-dimensional array to help select maximizing ojective
        objective_selected_dist = np.zeros(shape=(len(remaining_idx_set), 7))
        #find u \in train_active_mask\S maximizing phi'_u(S)
        start_epoch = timeit.default_timer()
        if len(sample_set )> 0:
            latest_added = sample_set[-1]
        for id , val in enumerate(remaining_idx_set):
            start_search = timeit.default_timer()
            #temp_set = sample_set.copy()
            temp_set = list(sample_set)
            #print("#######################")
            #print(id, val)
            ##y_km needs to be dynamically trimed
            pred_cluster_val = y_km[id]

            if cluster_flag_dict[pred_cluster_val]==False:
                #temp_set.add(val)
                temp_set.append(val)
                unprocess_first_term = non_negative_set(val, temp_set, typicality_dict, non_negative_set_dict)
                selected_sample_size = len(sample_set)
                if selected_sample_size not in non_negative_set_dict:
                    raise ValueError("The non_negative_set_dict doesn't save previous sample_size as key value!")
                non_negative_set_ind = unprocess_first_term - non_negative_set_dict[selected_sample_size]
                stop_search1 = timeit.default_timer()
                #print("Search Time 1:", stop_search1 - start_search)
                first_term = 0.5* non_negative_set_ind
                if len(sample_set )==0:
                    dist_ind = 0.0
                else:
                    dist_ind,dist_pair_array = optimized_dist(val, latest_added, sample_set, data_embeds,train_active_idx,euclid_dict, dist_pair_array)
                #dist_ind = dist(val, sample_set,data_embeds,train_active_idx,euclid_dict)
                objective = first_term + dist_ind*lam
                objective_selected_dist[id][0] = objective
                objective_selected_dist[id][1] = val
                objective_selected_dist[id][2] =pred_cluster_val
                objective_selected_dist[id][3] = id
                objective_selected_dist[id][4] = unprocess_first_term
                objective_selected_dist[id][5] = first_term
                objective_selected_dist[id][6] = dist_ind

                stop_search2 = timeit.default_timer()
                #print("Search Time 2:", stop_search2-start_search)
            else:
                continue
        ##sort the objective_selected_dist
        print("current sample size {:}".format(len(sample_set)))
        stop_epoch =timeit.default_timer()
        #print("Time:", stop_epoch-start_epoch)
        '''
        a =np.array([[9,2,3],
                  [4,5,5],[7,0,5]])
        b = [5]
        for ele in b:
            a =a[a[:,2]!=5,:]
        print a
        print(a[a[:,0].argsort()[::-1]])
        '''
        '''
        ##exclude the existing class
        for pred_c in range(np.max(y_km)+1):
            if cluster_flag_dict[pred_c]==True:
                objective_selected_dist =objective_selected_dist[objective_selected_dist[:,2]!=pred_c,:]
        '''
        objective_selected_dist = objective_selected_dist[objective_selected_dist[:,0].argsort()[::-1]]
        selected_idx = int(objective_selected_dist[0][1])
        #print("objective is {:.7f}".format(objective_selected_dist[0][0]))
        print("first_term is {:.7f}".format(objective_selected_dist[0][5]))
        print("second_term is {:.7f}".format(objective_selected_dist[0][6]))
        pred_class = int(objective_selected_dist[0][2])
        #sample_set.add(selected_idx)
        sample_set.append(selected_idx)
        ##update the non_negative_set_dict
        if len(sample_set) not in non_negative_set_dict:
            selected_sample_size = len(sample_set)
            non_negative_set_dict [selected_sample_size] = objective_selected_dist[0][4]
            ##update non_negative_set_dict dictionary for this updated sample size
            #non_negative_set_dict = update_non_negative_set_dict(selected_idx, sample_set, data_embeds, train_active_idx, euclid_dict, non_negative_set_dict)

        #print("add {} to sample_set".format(selected_idx))
        ##update the cluster_flag_dict
        if pred_class!=-1:
            cluster_flag_dict[pred_class] =True
        trimmed_id = int(objective_selected_dist[0][3])
        y_km = np.delete(y_km, trimmed_id, axis=0)
        ##reintialize cluster_flag_dict if all values ==True
        if all(value == True for value in cluster_flag_dict.values()) == True:
            for key in cluster_flag_dict.keys():
                if key not in set(y_km):
                    cluster_flag_dict.pop(key)
            cluster_flag_dict.update(dict(zip(cluster_flag_dict ,list(set(y_km)))))
            cluster_flag_dict = dict.fromkeys(cluster_flag_dict, False)

        #print(cluster_flag_dict.keys())

        #dynamic_data_embeds= np.delete(dynamic_data_embeds, np.where(dynamic_tain_active_idx  == selected_idx), axis=0)
        #dynamic_tain_active_idx =np.delete(dynamic_tain_active_idx,np.where(dynamic_tain_active_idx ==selected_idx))
    #already_selected = list(sample_set)
    already_selected = sample_set
    print("selected node id is {}".format(already_selected))
    train_active_mask[already_selected] = False
    assert (train_active_mask[train_active_mask == True].shape[0] == initial_true - sample_size)
    stop = timeit.default_timer()
    print("Time:", stop - start)
    return already_selected, train_active_mask


def un_approximation_sampler(cluster_size, sample_size, train_active_mask, data_embeds, ground_truth, lam):
    print("start implementing unoptimized approximation sampler")
    start = timeit.default_timer()
    print("calculating k-means first")
    km = KMeans(
        n_clusters=cluster_size, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    y_km = km.fit_predict(data_embeds)
    centroids = km.cluster_centers_
    x_dist = km.transform(data_embeds)

    y_km = km.fit_predict(data_embeds)
    centroids = km.cluster_centers_
    x_dist = km.transform(data_embeds)

    # print("initialize the empty set S")
    # sample_set = set()
    sample_set = []
    typicality_dict = dict()
    typicality_dict[-1] =0.0
    train_active_idx = np.where(train_active_mask == True)[0]
    initial_true = train_active_idx.shape[0]

    ##get a cluster_flage
    cluster_flag_dict = dict()

    for id, val in enumerate(set(train_active_idx.flatten())):
        ###get the corresponding euclidean dist to the centroid
        pred_cluster_val = y_km[id]
        cluster_flag_dict[pred_cluster_val] = False
        dist_val = x_dist[id][pred_cluster_val]
        # print("instance {0:} is assigned to {1:} cluster".format(val, pred_cluster_val))
        # print("euclidean distance is {:5.4f}".format(dist_val))
        if dist_val != 0.0:
            typicality_dict[val] = 1.0 / dist_val
        else:
            typicality_dict[val] = 1.0 / (dist_val + 1e-4)

    ##pre-compute the embedding distance between any two node-pair
    euclid_dict = dict()

    '''
    pre_computing_begin = timeit.default_timer()
    for i in range(len(train_active_idx)):
        for j in range(len(train_active_idx)):
            index_node_source = i
            index_node_dest = j
            #print(index_node_source)
            #print(index_node_dest)
            source_embeds = data_embeds[index_node_source]
            dest_embeds = data_embeds[index_node_dest]
            source_embeds = np.reshape(source_embeds, (1,-1))
            dest_embeds = np.reshape(dest_embeds, (1,-1))
            euclid_dict[(index_node_source, index_node_dest)] =euclidean_distances(source_embeds, dest_embeds)
    pre_computing_end = timeit.default_timer()
    print("Precomputing Time :", pre_computing_end -pre_computing_begin)
    '''

    non_negative_set_dict = dict()
    non_negative_set_dict[0] = 0.0
    ##initialize a n by len(sample_size) matrix to save dist computation
    dist_pair_array = np.zeros(shape=(len(train_active_idx), sample_size + 1))

    while len(sample_set) < sample_size:
        # get all the node index from train_active_mask\sample_set
        # train_active_idx = np.where(train_active_mask==True)[0]
        remaining_idx_set = set(train_active_idx.flatten()).difference(set(sample_set))
        # prepare a 2-dimensional array to help select maximizing ojective
        objective_selected_dist = np.zeros(shape=(len(remaining_idx_set), 7))
        # find u \in train_active_mask\S maximizing phi'_u(S)
        start_epoch = timeit.default_timer()
        if len(sample_set) > 0:
            latest_added = sample_set[-1]
        for id, val in enumerate(remaining_idx_set):
            start_search = timeit.default_timer()
            # temp_set = sample_set.copy()
            temp_set = list(sample_set)
            # print("#######################")
            # print(id, val)
            ##y_km needs to be dynamically trimed
            pred_cluster_val = y_km[id]

            if cluster_flag_dict[pred_cluster_val] == False:
                # temp_set.add(val)
                temp_set.append(val)
                unprocess_first_term = unoptimized_non_negative_set(val, temp_set, typicality_dict)
                #unprocess_first_term = non_negative_set(val, temp_set, typicality_dict, non_negative_set_dict)
                selected_sample_size = len(sample_set)
                if selected_sample_size not in non_negative_set_dict:
                    raise ValueError("The non_negative_set_dict doesn't save previous sample_size as key value!")
                #non_negative_set_ind = unprocess_first_term - non_negative_set_dict[selected_sample_size]
                non_negative_set_ind =  unprocess_first_term - unoptimized_non_negative_set(-1, temp_set, typicality_dict)
                stop_search1 = timeit.default_timer()
                # print("Search Time 1:", stop_search1 - start_search)
                first_term = 0.5 * non_negative_set_ind
                if len(sample_set) == 0:
                    dist_ind = 0.0
                else:
                    '''
                    dist_ind, dist_pair_array = optimized_dist(val, latest_added, sample_set, data_embeds,
                                                               train_active_idx, euclid_dict, dist_pair_array)
                    '''
                    dist_ind = dist(val, sample_set,data_embeds,train_active_idx,euclid_dict)
                objective = first_term + dist_ind * lam
                objective_selected_dist[id][0] = objective
                objective_selected_dist[id][1] = val
                objective_selected_dist[id][2] = pred_cluster_val
                objective_selected_dist[id][3] = id
                objective_selected_dist[id][4] = unprocess_first_term
                objective_selected_dist[id][5] = first_term
                objective_selected_dist[id][6] = dist_ind

                stop_search2 = timeit.default_timer()
                # print("Search Time 2:", stop_search2-start_search)
            else:
                continue
        ##sort the objective_selected_dist
        print("current sample size {:}".format(len(sample_set)))
        stop_epoch = timeit.default_timer()
        # print("Time:", stop_epoch-start_epoch)
        '''
        a =np.array([[9,2,3],
                  [4,5,5],[7,0,5]])
        b = [5]
        for ele in b:
            a =a[a[:,2]!=5,:]
        print a
        print(a[a[:,0].argsort()[::-1]])
        '''
        '''
        ##exclude the existing class
        for pred_c in range(np.max(y_km)+1):
            if cluster_flag_dict[pred_c]==True:
                objective_selected_dist =objective_selected_dist[objective_selected_dist[:,2]!=pred_c,:]
        '''
        objective_selected_dist = objective_selected_dist[objective_selected_dist[:, 0].argsort()[::-1]]
        selected_idx = int(objective_selected_dist[0][1])
        # print("objective is {:.7f}".format(objective_selected_dist[0][0]))
        print("first_term is {:.7f}".format(objective_selected_dist[0][5]))
        print("second_term is {:.7f}".format(objective_selected_dist[0][6]))
        pred_class = int(objective_selected_dist[0][2])
        # sample_set.add(selected_idx)
        sample_set.append(selected_idx)
        ##update the non_negative_set_dict
        if len(sample_set) not in non_negative_set_dict:
            selected_sample_size = len(sample_set)
            non_negative_set_dict[selected_sample_size] = objective_selected_dist[0][4]
            ##update non_negative_set_dict dictionary for this updated sample size
            # non_negative_set_dict = update_non_negative_set_dict(selected_idx, sample_set, data_embeds, train_active_idx, euclid_dict, non_negative_set_dict)

        # print("add {} to sample_set".format(selected_idx))
        ##update the cluster_flag_dict
        if pred_class != -1:
            cluster_flag_dict[pred_class] = True
        trimmed_id = int(objective_selected_dist[0][3])
        y_km = np.delete(y_km, trimmed_id, axis=0)
        ##reintialize cluster_flag_dict if all values ==True
        if all(value == True for value in cluster_flag_dict.values()) == True:
            for key in cluster_flag_dict.keys():
                if key not in set(y_km):
                    cluster_flag_dict.pop(key)
            cluster_flag_dict.update(dict(zip(cluster_flag_dict, list(set(y_km)))))
            cluster_flag_dict = dict.fromkeys(cluster_flag_dict, False)

        # print(cluster_flag_dict.keys())

        # dynamic_data_embeds= np.delete(dynamic_data_embeds, np.where(dynamic_tain_active_idx  == selected_idx), axis=0)
        # dynamic_tain_active_idx =np.delete(dynamic_tain_active_idx,np.where(dynamic_tain_active_idx ==selected_idx))
    # already_selected = list(sample_set)
    already_selected = sample_set
    print("selected node id is {}".format(already_selected))
    train_active_mask[already_selected] = False
    assert (train_active_mask[train_active_mask == True].shape[0] == initial_true - sample_size)
    stop = timeit.default_timer()
    print("Time:", stop - start)
    return already_selected, train_active_mask













    '''
    # plot the 3 clusters
    plt.scatter(
        data_embeds[y_km == 0, 0], data_embeds[y_km == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )

    plt.scatter(
        data_embeds[y_km == 1, 0], data_embeds[y_km == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )

    plt.scatter(
        data_embeds[y_km == 2, 0], data_embeds[y_km == 2, 1],
        s=50, c='lightblue',
        marker='v', edgecolor='black',
        label='cluster 3'
    )

    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()
    '''
    '''
    n, dim = data_embeds.shape
    kmeans = faiss.Kmeans(dim, sample_size, niter=20, nredo=5, verbose=True)
    kmeans.train(data_embeds)
    centroids_2 = kmeans.centroids




    index = faiss.IndexFlatL2(dim)
    '''
    '''
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    '''


    '''
    index.add(centroids_2)
    D,I = index.search(centroids_2,2)
    idx_range = np.where(train_active_mask == True)[0]
    initial_true = idx_range.shape[0]



    
    clustermembers.extend(np.where(assignments == i)[0] for i in range(budget))

    # find k nearest neighbors to each centroid
    index = faiss.IndexFlatL2(dim)
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(inference_feats)
    k = 200
    _, nn = gpu_index.search(centroids, k)

    empty_clusters = 0
    valid_nn = np.ones(budget, dtype=np.int32) * -1
    for i in tqdm(range(budget), desc="valid nn:"):
        valid_members = np.array(nn[i])[np.in1d(np.array(nn[i]), np.array(clustermembers[i]))]
        if len(valid_members):
            valid_nn[i] = valid_members[0]

        if not len(np.array(clustermembers[i])):
            empty_clusters += 1

    assert len(np.where(valid_nn == -1)[
                   0]) == empty_clusters, f'knn is small: {empty_clusters} empty clusters, {len(np.where(valid_nn == -1)[0])} clusters without nn'
    valid_nn = np.delete(valid_nn, np.where(valid_nn == -1)[0])
    '''


def load_unprocessed_data(datapath,datasetname):

    print("start evaluating twitter dataset!")
    attr_filename = datapath + datasetname+'/'+'Attribute'
    lines =[]
    with open(attr_filename) as f:
        lines = f.readlines()
    count = 0
    line_num =len(lines)
    attr_dict =dict()
    attr_id_map = dict()
    for line in lines:
        print('line {}:{}'.format(count, line))
        item = line.split('\t')
        attr_id = int(item[0])
        attr_name = item[1].split(':')[0]
        attr_value = item[1].split(':')[1].split('\n')[0]
        print('attr_name {} ///// attr_value {}'.format(attr_name, attr_value))
        if attr_name not in attr_dict:
            attr_dict[attr_name] = []
            attr_dict[attr_name].append([attr_value, attr_id])
        else:
            attr_dict[attr_name].append([attr_value, attr_id])
        if attr_id not in attr_id_map:
            attr_id_map[attr_id]=[attr_name, attr_value]
        else:
            raise ValueError
        count+=1
    assert count == line_num
    print("finish building the dict of attrs")

    ##get how many nodes from this dataset
    statistics_filename = datapath + datasetname + '/' + 'statistics'
    lines = []
    with open(statistics_filename) as f:
        lines = f.readlines()
    num_nodes = int(lines[0].split('\n')[0])

    ##build an ndarray to store the attribute for each node
    col_names = []
    for attr in attr_dict.keys():
        col_names.append(attr)
    attr_array = np.zeros(shape=(num_nodes, len(col_names)))
    attr_array = attr_array.astype(str)
    attr_array = np.where(attr_array=='0.0','NA','0.0')
    attr_array = attr_array.astype(object)



    ##build the data table for the dataset
    print("start building the csv file for {} dataset".format(datasetname))
    print("reading the vertex2aid file")
    data_table = datapath + datasetname + '/' + 'vertex2aid'
    lines = []
    with open(data_table) as f:
        lines = f.readlines()
    count = 0
    line_num = len(lines)
    for line in lines:
        print('line {}:{}'.format(count, line))
        item = line.split('\t')
        node_id = int(item[0])
        attr_id = int(item[1])
        ##according to attr_id locate which column and set the corresponding value
        row_idx = node_id
        col_idx = col_names.index(attr_id_map[attr_id][0])
        attr_array[row_idx, col_idx] = attr_id_map[attr_id][1]
        count += 1
    assert count == line_num
    print("finish building the attrs ndarray")
    ##postprocessing removes the majority of the N.A.
    row_rmv_idx =[]
    for i in range(attr_array.shape[0]):
        count =0
        for j in range(attr_array.shape[1]):
            if attr_array[i][j]=='NA':
                count+=1
            if count>2:
                row_rmv_idx.append(i)
                break
    print("removing the row_rmv_idx")
    attr_array = np.delete(attr_array,row_rmv_idx,axis=0)




    ##save the attrs ndarrya to a csv file for constraint development
    dataframe = pd.DataFrame(attr_array, columns=col_names)
    csv_filename = datapath + datasetname + '/' + datasetname+'.csv'
    dataframe.to_csv(csv_filename,index=False)


def non_negative_set(val, active_set, typicality_dict, non_negative_set_dict):
    saved_active_set_key = len(active_set)
    if saved_active_set_key not in non_negative_set_dict:
        saved_active_set_key -=1
        if saved_active_set_key not in non_negative_set_dict:
            raise ValueError("The non_negative_set_dict doesn't save previous sample_size as key value!")
    sum_dist = non_negative_set_dict[saved_active_set_key]

    selected_idx =val
    dist = typicality_dict[selected_idx]
    sum_dist += dist
    return sum_dist

def unoptimized_non_negative_set(val, active_set, typicality_dict):
    sum_dist =0.0
    for ele in active_set:
        sum_dist +=typicality_dict[ele]
    sum_dist+= typicality_dict[val]
    return sum_dist



def dist(val, sample_set,data_embeds,train_active_idx, euclid_dict):
    euclid_dist =0.0
    row_index = int(np.where(train_active_idx==val)[0])

    sample_embeds = data_embeds[row_index]
    sample_embeds = np.reshape(sample_embeds, (1,-1))
    for selected_ele in sample_set:
        idx = int(np.where(train_active_idx==selected_ele)[0])
        selected_embeds = data_embeds[idx]
        selected_embeds = np.reshape(selected_embeds, (1,-1))
        #print(euclidean_distances(sample_embeds ,selected_embeds))
        euclid_dist +=euclidean_distances(sample_embeds ,selected_embeds)
        #euclid_dist +=euclid_dict[(row_index,idx)]
    return euclid_dist


def update_non_negative_set_dict(selected_idx, sample_set, data_embeds, train_active_idx, euclid_dict, non_negative_set_dict):
    saved_sample_size_key = len(sample_set)-1
    if saved_sample_size_key not in non_negative_set_dict:
        raise ValueError("The non_negative_set_dict doesn't save previous sample_size as key value!")

    euclid_dist =0.0
    row_index = int(np.where(train_active_idx==selected_idx)[0])

    sample_embeds = data_embeds[row_index]
    sample_embeds = np.reshape(sample_embeds, (1,-1))
    idx = int(np.where(train_active_idx==selected_idx)[0])
    selected_embeds = data_embeds[idx]
    selected_embeds = np.reshape(selected_embeds, (1,-1))
    #print(euclidean_distances(sample_embeds ,selected_embeds))
    euclid_dist +=euclidean_distances(sample_embeds ,selected_embeds)
    #euclid_dist +=euclid_dict[(row_index,idx)]


def optimized_dist(val, latest_added, sample_set, data_embeds,train_active_idx,euclid_dict, dist_pair_array):
    saved_sample_column_indx = len(sample_set)
    row_index = int(np.where(train_active_idx == val)[0])
    euclid_dist=dist_pair_array[row_index, saved_sample_column_indx]

    sample_embeds = data_embeds[row_index]
    sample_embeds = np.reshape(sample_embeds, (1, -1))
    selected_ele = latest_added
    idx = int(np.where(train_active_idx == selected_ele)[0])
    selected_embeds = data_embeds[idx]
    selected_embeds = np.reshape(selected_embeds, (1, -1))
    # print(euclidean_distances(sample_embeds ,selected_embeds))
    euclid_dist += euclidean_distances(sample_embeds, selected_embeds)
    # euclid_dist +=euclid_dict[(row_index,idx)]
    ##update dist_pair_array
    dist_pair_array[row_index, saved_sample_column_indx+1] = euclid_dist
    return euclid_dist, dist_pair_array


def cmpt_page_rank(pr_prob, adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])).tocsr()
    page_rank_matrix = pr_prob * (inv((sp.eye(adj.shape[0]) - (1 - pr_prob) * adj_normalized).tocsc()))
    return page_rank_matrix


def topological_typi(dense_page_rank, soft_label_matrix, train_active_mask,dist_active_data,train_active_idx,node_id, node_id_index):

    count_error_label = 0
    count_correct_label =0
    P_error=[]
    P_correct =[]
    for id in range(len(train_active_idx)):
        n_id = train_active_idx[id]
        #print(np.argmax(soft_label_matrix[n_id]))
        #print(np.argmax(dist_active_data[id]))
        if np.argmax(soft_label_matrix[n_id]) != np.argmax(dist_active_data[id]) and np.argmax(dist_active_data[id])==0:
            count_correct_label +=1
            P_correct.append(dense_page_rank[node_id][id])
        elif np.argmax(soft_label_matrix[n_id]) != np.argmax(dist_active_data[id]) and np.argmax(dist_active_data[id])==1:
            count_error_label += 1
            P_error.append(dense_page_rank[node_id][id])
        else:
            continue
    #weighted sum for Totoro metric computation
    if len(P_correct)> 0 and len(P_error)> 0:
        Totoro = 1.0/count_correct_label *np.sum(P_correct) +1.0/count_error_label *np.sum(P_error)
    elif len(P_correct)> 0 and len(P_error)<=0:
        Totoro = 1.0 / count_correct_label * np.sum(P_correct)
    elif len(P_error)> 0 and len(P_correct)<= 0:
        Totoro =1.0/count_error_label *np.sum(P_error)
    else:
        Totoro =0.0
    topological_typicality = 1 -Totoro
    return topological_typicality





def soft_approximation_sampler(cluster_size, sample_size, train_active_mask, data_embeds, ground_truth, lam, dense_page_rank, soft_label_matrix, dist_active_data):

    print("start implementing approximation sampler")
    start = timeit.default_timer()
    print("calculating k-means first")
    km = KMeans(
        n_clusters=cluster_size, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    y_km = km.fit_predict(data_embeds)
    centroids = km.cluster_centers_
    x_dist = km.transform(data_embeds)


    #print("initialize the empty set S")
    #sample_set = set()
    sample_set = []
    typicality_dict = dict()
    train_active_idx = np.where(train_active_mask == True)[0]
    initial_true =  train_active_idx.shape[0]
    #print(data_embeds.shape[0])

    ##get a cluster_flage
    cluster_flag_dict = dict()



    for id, val in enumerate(set(train_active_idx.flatten())):
        ###get the corresponding euclidean dist to the centroid
        pred_cluster_val = y_km[id]
        cluster_flag_dict[pred_cluster_val] = False
        dist_val = x_dist[id][pred_cluster_val]
        #print("instance {0:} is assigned to {1:} cluster".format(val, pred_cluster_val))
        #print("euclidean distance is {:5.4f}".format(dist_val))
        train_id =val
        train_id_index = id

        if dist_val !=0.0:
            typicality_dict[val] = 1.0/dist_val * topological_typi(dense_page_rank, soft_label_matrix, train_active_mask,dist_active_data,train_active_idx, train_id, train_id_index)
        else:
            typicality_dict[val] = 1.0/(dist_val+1e-4) * topological_typi(dense_page_rank, soft_label_matrix, train_active_mask,dist_active_data,train_active_idx, train_id, train_id_index)

    print("computing typicality finished")
    ##pre-compute the embedding distance between any two node-pair
    euclid_dict = dict()

    '''
    pre_computing_begin = timeit.default_timer()
    for i in range(len(train_active_idx)):
        for j in range(len(train_active_idx)):
            index_node_source = i
            index_node_dest = j
            #print(index_node_source)
            #print(index_node_dest)
            source_embeds = data_embeds[index_node_source]
            dest_embeds = data_embeds[index_node_dest]
            source_embeds = np.reshape(source_embeds, (1,-1))
            dest_embeds = np.reshape(dest_embeds, (1,-1))
            euclid_dict[(index_node_source, index_node_dest)] =euclidean_distances(source_embeds, dest_embeds)
    pre_computing_end = timeit.default_timer()
    print("Precomputing Time :", pre_computing_end -pre_computing_begin)
    '''



    #dynamic_tain_active_idx = train_active_idx.copy()
    #dynamic_data_embeds = data_embeds.copy()


    non_negative_set_dict = dict()
    non_negative_set_dict[0]= 0.0
    ##initialize a n by len(sample_size) matrix to save dist computation
    dist_pair_array = np.zeros(shape=(len(train_active_idx), sample_size+1))

    while len(sample_set)< sample_size:
        #get all the node index from train_active_mask\sample_set
        #train_active_idx = np.where(train_active_mask==True)[0]
        remaining_idx_set = set(train_active_idx.flatten()).difference(set(sample_set))
        # prepare a 2-dimensional array to help select maximizing ojective
        objective_selected_dist = np.zeros(shape=(len(remaining_idx_set), 7))
        #find u \in train_active_mask\S maximizing phi'_u(S)
        start_epoch = timeit.default_timer()
        if len(sample_set )> 0:
            latest_added = sample_set[-1]
        for id , val in enumerate(remaining_idx_set):
            start_search = timeit.default_timer()
            #temp_set = sample_set.copy()
            temp_set = list(sample_set)
            #print("#######################")
            #print(id, val)
            ##y_km needs to be dynamically trimed
            pred_cluster_val = y_km[id]

            if cluster_flag_dict[pred_cluster_val]==False:
                #temp_set.add(val)
                temp_set.append(val)
                unprocess_first_term = non_negative_set(val, temp_set, typicality_dict, non_negative_set_dict)
                selected_sample_size = len(sample_set)
                if selected_sample_size not in non_negative_set_dict:
                    raise ValueError("The non_negative_set_dict doesn't save previous sample_size as key value!")
                non_negative_set_ind = unprocess_first_term - non_negative_set_dict[selected_sample_size]
                stop_search1 = timeit.default_timer()
                #print("Search Time 1:", stop_search1 - start_search)
                first_term = 0.5* non_negative_set_ind
                if len(sample_set )==0:
                    dist_ind = 0.0
                else:
                    dist_ind,dist_pair_array = optimized_dist(val, latest_added, sample_set, data_embeds,train_active_idx,euclid_dict, dist_pair_array)
                #dist_ind = dist(val, sample_set,data_embeds,train_active_idx,euclid_dict)
                objective = first_term + dist_ind*lam
                objective_selected_dist[id][0] = objective
                objective_selected_dist[id][1] = val
                objective_selected_dist[id][2] =pred_cluster_val
                objective_selected_dist[id][3] = id
                objective_selected_dist[id][4] = unprocess_first_term
                objective_selected_dist[id][5] = first_term
                objective_selected_dist[id][6] = dist_ind

                stop_search2 = timeit.default_timer()
                #print("Search Time 2:", stop_search2-start_search)
            else:
                continue
        ##sort the objective_selected_dist
        print("current sample size {:}".format(len(sample_set)))
        stop_epoch =timeit.default_timer()
        #print("Time:", stop_epoch-start_epoch)
        '''
        a =np.array([[9,2,3],
                  [4,5,5],[7,0,5]])
        b = [5]
        for ele in b:
            a =a[a[:,2]!=5,:]
        print a
        print(a[a[:,0].argsort()[::-1]])
        '''
        '''
        ##exclude the existing class
        for pred_c in range(np.max(y_km)+1):
            if cluster_flag_dict[pred_c]==True:
                objective_selected_dist =objective_selected_dist[objective_selected_dist[:,2]!=pred_c,:]
        '''
        objective_selected_dist = objective_selected_dist[objective_selected_dist[:,0].argsort()[::-1]]
        selected_idx = int(objective_selected_dist[0][1])
        #print("objective is {:.7f}".format(objective_selected_dist[0][0]))
        print("first_term is {:.7f}".format(objective_selected_dist[0][5]))
        print("second_term is {:.7f}".format(objective_selected_dist[0][6]))
        pred_class = int(objective_selected_dist[0][2])
        #sample_set.add(selected_idx)
        sample_set.append(selected_idx)
        ##update the non_negative_set_dict
        if len(sample_set) not in non_negative_set_dict:
            selected_sample_size = len(sample_set)
            non_negative_set_dict [selected_sample_size] = objective_selected_dist[0][4]
            ##update non_negative_set_dict dictionary for this updated sample size
            #non_negative_set_dict = update_non_negative_set_dict(selected_idx, sample_set, data_embeds, train_active_idx, euclid_dict, non_negative_set_dict)

        #print("add {} to sample_set".format(selected_idx))
        ##update the cluster_flag_dict
        if pred_class!=-1:
            cluster_flag_dict[pred_class] =True
        trimmed_id = int(objective_selected_dist[0][3])
        y_km = np.delete(y_km, trimmed_id, axis=0)
        ##reintialize cluster_flag_dict if all values ==True
        if all(value == True for value in cluster_flag_dict.values()) == True:
            for key in cluster_flag_dict.keys():
                if key not in set(y_km):
                    cluster_flag_dict.pop(key)
            cluster_flag_dict.update(dict(zip(cluster_flag_dict ,list(set(y_km)))))
            cluster_flag_dict = dict.fromkeys(cluster_flag_dict, False)

        #print(cluster_flag_dict.keys())

        #dynamic_data_embeds= np.delete(dynamic_data_embeds, np.where(dynamic_tain_active_idx  == selected_idx), axis=0)
        #dynamic_tain_active_idx =np.delete(dynamic_tain_active_idx,np.where(dynamic_tain_active_idx ==selected_idx))
    #already_selected = list(sample_set)
    already_selected = sample_set


    ##update soft_label_matrix!!


    print("selected node id is {}".format(already_selected))
    train_active_mask[already_selected] = False
    assert (train_active_mask[train_active_mask == True].shape[0] == initial_true - sample_size)
    stop = timeit.default_timer()
    print("Time:", stop - start)
    return already_selected, train_active_mask




























