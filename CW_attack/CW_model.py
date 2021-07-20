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
                 attack_strength, alpha, para_c, ori_model):
        self.graph = graph
        self.call_relation = call_relation
        self.state_idx = state_idx
        self.substitute = trained_model
        self.K = attack_strength
        self.NUM_STATE = 11
        self.alpha = alpha
        self.c = para_c
        self.ori_model = ori_model

    def step(self, call_relation, w):
        grad, loss = self.cal_gradient(call_relation, w)
        w = w + torch.relu(self.alpha * grad)
        return w, loss

    def cal_gradient(self, x_adv, w):
        w_tmp = Variable(w, requires_grad=True)
        call_relation = x_adv + w_tmp
        MarkovFeats = torch.zeros((self.NUM_STATE, self.NUM_STATE))
        Norma_all = torch.sum(call_relation, axis=1)
        for i in range(0, len(call_relation)):
            Norma = Norma_all[i]
            if Norma == 0:
                MarkovFeats[i] = call_relation[i]
            else:
                MarkovFeats[i] = call_relation[i] / Norma
        feature = MarkovFeats.flatten()
        feature = torch.reshape(feature, (1, -1))
        z_x_t = self.substitute(feature)
        tmp = torch.cat((torch.reshape(0 - z_x_t[0][1], (1, -1)), torch.reshape(torch.tensor(-self.K), (1, -1))))
        f = torch.max(tmp)
        loss = torch.sum(torch.sum(torch.square(w_tmp))) + self.c * f  # line 3
        _gradient = loss.backward()
        return w_tmp.grad, loss

    def cal_loss(self, x_adv, w):
        feature = self.extract_feature(x_adv + w)
        z_x_t = self.substitute(feature.reshape(1, -1))
        tmp = torch.cat((torch.reshape(0 - z_x_t[0][1], (1, -1)), torch.reshape(torch.tensor(-self.K), (1, -1))))
        f = torch.max(tmp)
        loss = torch.sum(torch.sum(torch.square(w))) + self.c * f
        return loss
