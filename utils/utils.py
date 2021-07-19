# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connet dot polyu dot hk
# datetime: 2021/7/20 12:56 AM
# software: PyCharm
import numpy as np
import torch


def knn_predict_torch(data, label, k, test_data):
    num_data = np.arrange(label.shape[0])
    dist = torch.sum((np.squeeze(data) - np.squeeze(test_data)).pow(2.0), 1)
    ind = torch.argsort(dist)
    predict_label = label[ind[:k]]
    if k == 1:
        return predict_label
    else:
        unique_label = torch.unique(predict_label)
        unique_label = unique_label.long()
        count = np.zeros(unique_label.shape[0])
        for i in label:
            count[unique_label[i.long()]] += 1
        ii = torch.argmax(torch.from_numpy(count))
        return unique_label[ii]
