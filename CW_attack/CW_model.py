# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connet dot polyu dot hk
# datetime: 2021/7/19 4:47 PM
# software: PyCharm
import numpy as np
import torch
from torch.autograd import Variable


class CW_Attack(object):
    def __init__(self, graph, call_relation, state_idx, trained_model,
                 attack_strength, alpha):
        self.graph = graph
        self.call_relation = call_relation
        self.state_idx = state_idx
        self.substitute = trained_model
        self.K = attack_strength
        self.NUM_STATE = 11
        self.alpha = alpha

    def step(self, call_relation, w):
        grad = self.cal_gradient(call_relation, w)
        w = w + torch.clip(self.alpha * grad)
        return w


    def cal_gradient(self, x_adv, w):
        w_tmp = Variable(w)
        loss = self.cal_loss(x_adv, w_tmp)
        gradient = loss.backward()
        return gradient

    def cal_loss(self, x_adv, w):
        feature = self.extract_feature(x_adv + w)
        z_x_t = self.substitute(feature)
        loss = torch.max(1 - z_x_t[1], -self.K)
        return loss

    def extract_feature(self, call_relation):
        # torch.zeros()
        MarkovFeats = torch.zeros((self.NUM_STATE, self.NUM_STATE))
        Norma_all = torch.sum(call_relation, axis=1)
        for i in range(0, len(call_relation)):
            Norma = Norma_all[i]
            if Norma == 0:
                MarkovFeats[i] = call_relation[i]
            else:
                MarkovFeats[i] = call_relation[i] / Norma

        feature = MarkovFeats.flatten()
        return feature
