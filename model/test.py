# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connet dot polyu dot hk
# datetime: 2021/7/12 5:55 PM
# software: PyCharm
import pickle
data_path = "/Users/zhaokaifa/OneDrive - The Hong Kong Polytechnic University/Code/myattack_QL_mac_mamadroid"

df = open(data_path+"/preprocess/test_csr_dict_families_benign.pkl", "rb")
data = pickle.load(df)
test_sha256_all = data["sha256"]
test_adj_all = data["adjacent_matrix"]
test_idx_all = data["node_idx"]
a = 1