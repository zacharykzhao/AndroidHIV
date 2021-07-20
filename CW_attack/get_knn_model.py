# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connet dot polyu dot hk
# datetime: 2021/7/20 1:06 AM
# software: PyCharm


from sklearn.neighbors import KNeighborsClassifier

import pickle

from CW_attack.attack import load_train

data_path = "/Users/zhaokaifa/OneDrive - The Hong Kong Polytechnic University/Code/myattack_QL_mac_mamadroid"
train_feature, train_label = load_train(data_path + "/preprocess/train_csr_dict_families.pkl")

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(train_feature, train_label)
pickle.dump(knn1, open('knn1_2021.pkl', 'wb'))
aaa = knn1.predict_proba(train_feature[0].reshape(1,-1))
b = 1