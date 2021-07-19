# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connet dot polyu dot hk
# datetime: 2021/7/19 4:51 PM
# software: PyCharm
import numpy as np

NUM_STATE = 11


def get_call_number(graph, state_idx):
    '''
    for given apk (FCG), obtain the call number between states
    :param graph: adjacent martrix
    :param state_idx:
    :return:
    '''
    idx_one_hot = np.zeros((state_idx.size, NUM_STATE))
    idx_one_hot[np.arange(state_idx.size), state_idx] = 1
    #
    call_relation = idx_one_hot.transpose().dot(graph.dot(idx_one_hot))
    return call_relation


def get_feature(graph, state_idx, call_number):
    '''
    given apk (FCG) and the call number between states, obtain feature vector
    :param graph:
    :param state_idx:
    :param call_number:
    :return:
    '''

    call_relation = get_call_number(graph, state_idx)
    return extract_feature(call_relation)


def extract_feature(call_relation):
    MarkovFeats = np.zeros((NUM_STATE, NUM_STATE))
    Norma_all = np.sum(call_relation, axis=1)
    for i in range(0, len(call_relation)):
        Norma = Norma_all[i]
        if Norma == 0:
            MarkovFeats[i] = call_relation[i]
        else:
            MarkovFeats[i] = call_relation[i] / Norma

    feature = MarkovFeats.flatten()
    return feature


def get_perturb(call_relation, call_perturb):
    perturb = np.zeros((NUM_STATE, NUM_STATE))
    norm_ori = np.sum(call_relation, axis=1)
    norm_call_perturb = np.sum(call_perturb, axis=1)
    count = 0
    for i in range(0, len(call_relation)):
        no_i = norm_ori[i]
        no_p = norm_call_perturb[i]
        if no_i == 0:
            perturb[count] = np.zeros((1, NUM_STATE))
        else:
            perturb[count] = (call_relation[i] + call_perturb[i]) / (no_i + no_p) - call_relation[i] / no_i

    return perturb
