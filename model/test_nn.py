# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connet dot polyu dot hk
# datetime: 2021/7/12 8:44 PM
# software: PyCharm
import pickle
import torch
import numpy as np

from model.libs import extract_feature, find_nn_torch

data_path = "/Users/zhaokaifa/OneDrive - The Hong Kong Polytechnic University/Code/myattack_QL_mac_mamadroid"

if __name__ == '__main__':
    ## load train data
    df = open(data_path + "/preprocess/train_csr_dict_families.pkl", "rb")
    train_data = pickle.load(df)
    train_feature = train_data["data"]
    train_label = train_data["label"]

    d_feature = len(train_feature[0])

    ## preprocess data
    train_feature = torch.from_numpy(np.array(train_feature)).float()
    train_label = torch.from_numpy(np.array(train_label))
    # load test data
    # df = open(data_path+"/preprocess/test_csr_dict_families.pkl", "rb")
    df = open("/Users/zhaokaifa/OneDrive - The Hong Kong Polytechnic University/Code/A2DataPreprocess/"
              "algorithm_evaluation/evaluateMamadroid/test_csr_dict_2011_families.pkl","rb")

    data = pickle.load(df)
    test_sha256_all = data["sha256"]
    test_adj_all = data["adjacent_matrix"]
    test_idx_all = data["node_idx"]

    for zidx, test_sha256 in enumerate(test_sha256_all):

        test_adj = test_adj_all[zidx]
        pack_idx = test_idx_all[zidx]
        idx_one_hot = np.zeros((pack_idx.size, 11))
        idx_one_hot[np.arange(pack_idx.size), pack_idx] = 1

        test_feature = extract_feature(test_adj, pack_idx)
        test_feature = torch.from_numpy(np.array(test_feature)[np.newaxis,:]).float()
        predict = find_nn_torch(test_feature,train_feature, train_label)
        print(test_sha256,":\t", predict)
