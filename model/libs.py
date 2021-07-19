# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connet dot polyu dot hk
# datetime: 2021/7/12 7:35 PM
# software: PyCharm

import numpy as np
import torch


def extract_feature(adj, pack_idx, type="families"):
    '''
    adj: csr_matrix: adjacent matrix
    pack_idx: nparray: node number * 1, the package index of each node
    num: package: 11; package: 446
    '''
    if type == "families":
        nn = 11
    else:
        nn = 446
    idx_one_hot = np.zeros((pack_idx.size, nn))

    idx_one_hot[np.arange(pack_idx.size), pack_idx] = 1

    #
    call_relation = idx_one_hot.transpose().dot(adj.dot(idx_one_hot))

    # MarkovFeats = np.zeros((max(pack_idx)+1, max(pack_idx)+1))
    MarkovFeats = np.zeros((nn, nn))
    tmpaa = []
    Norma_all = np.sum(call_relation, axis=1)
    for i in range(0, len(call_relation)):
        Norma = Norma_all[i]
        tmpaa.append(Norma)
        if Norma == 0:
            MarkovFeats[i] = call_relation[i]
        else:
            MarkovFeats[i] = call_relation[i] / Norma

    feature = MarkovFeats.flatten()
    return feature



def find_nn_torch(Q, X, y, k=1):
    '''
    :param Q: target data
    :param X:  training data
    :param y:  training label
    :param k: neighbor
    :return:
    '''
    xsize = np.arange(y.shape[0])
    np.random.shuffle(xsize)
    X = X[xsize[0:1000], :]
    y = y[xsize[0:1000]]
    dist = torch.sum((np.squeeze(X) - np.squeeze(Q)).pow(2.), 1)
    ind = torch.argsort(dist)
    label = y[ind[:k]]
    unique_label = torch.unique(y)
    unique_label = unique_label.long()
    count = np.zeros(unique_label.shape[0])
    for i in label:
        count[unique_label[i.long()]] += 1
    ii = torch.argmax(torch.from_numpy(count))
    final_label = unique_label[ii]
    return final_label