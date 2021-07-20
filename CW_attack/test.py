# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connet dot polyu dot hk
# datetime: 2021/7/20 1:38 AM
# software: PyCharm
import os

for i in range(10):
    f = open("test.txt", "a")
    f.write(str(i))
    f.write('\n')