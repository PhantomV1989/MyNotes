# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 23:38:21 2017

@author: PhantomV
"""

import numpy as np
import copy

from rbm import RBM



a=[np.random.rand(5)]

xx=RBM(0.02,hidden_size=3,CDk=100,n_epochs=1)
xx.train(a)

outputstr=''
for i in a[0]:
    outputstr+=str(i)+','
outputstr+='\n'
for i in xx.px_list:
    for j in range(len(i)):
        outputstr+=str(i[j])
        if j<len(i)-1:outputstr+=','
    outputstr+='\n'
    

with open("Output.csv", "w") as text_file:
    text_file.write(outputstr)