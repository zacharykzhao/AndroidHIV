# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connet dot polyu dot hk
# datetime: 2021/7/12 4:57 PM
# software: PyCharm

'''
    Settings from Android-HIV:
        optimization algorithm: AdaGrad
        batch size: 256
        epoch: 100
'''

import pickle

import torch
import torch.optim as optim
## hyper-parameter
from model.Substitute_model import Substitute
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
BATCH_SIZE = 256
EPOCHS = 100

data_path = "/Users/zhaokaifa/OneDrive - The Hong Kong Polytechnic University/Code/myattack_QL_mac_mamadroid"
if __name__ == '__main__':
    ## load data
    df = open(data_path + "/preprocess/train_csr_dict_families.pkl", "rb")
    train_data = pickle.load(df)
    train_feature = train_data["data"]
    train_label = train_data["label"]

    d_feature = len(train_feature[0])

    ## preprocess data
    train_feature = torch.from_numpy(np.array(train_feature)).float()
    train_label = torch.from_numpy(np.array(train_label))
    torch_dataset = Data.TensorDataset(train_feature, train_label)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=2,
    )

    ##
    net = Substitute(d_feature)
    optimizer = optim.Adagrad(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # train
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            # batch_y = torch.nn.functional.one_hot(batch_y, 2)
            # forward + backward + optimize
            outputs = net(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            if np.isnan(loss.item()):
                print('Loss value is NaN!')
            # print statistics
            running_loss += loss.item()
            if step % 50 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %f' %
                      (epoch + 1, step + 1, running_loss / 2000))
                running_loss = 0.0
    PATH = './substitute_model.pth'
    torch.save(net.state_dict(), PATH)


