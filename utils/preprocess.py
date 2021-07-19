# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connet dot polyu dot hk
# datetime: 2021/7/19 4:51 PM
# software: PyCharm
import numpy as np


def get_call_number(graph, state_idx):
    '''
    for given apk (FCG), obtain the call number between states
    :param graph: adjacent martrix
    :param state_idx:
    :return:
    '''
    no_state = 11

    idx_one_hot = np.zeros((state_idx.size, no_state))

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


