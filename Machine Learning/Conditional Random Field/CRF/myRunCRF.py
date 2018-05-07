# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:49:19 2017

@author: PhantomV
"""

from crf import LinearChainCRF
import numpy as np
import os
import sys
#import fcntl
import copy
from string import Template
import mlpython.datasets.store as dataset_store
import mlpython.mlproblems.generic as mlpb
import pdb


class SampleData:
    def __init__(self):
        print ("Loading dataset...")
        import ocr_letters_sequential as mldataset
        load_to_memory = True
        
        all_data = mldataset.load("",load_to_memory=load_to_memory)
        
        self.train_data, self.train_metadata = all_data['train']
        self.valid_data, self.valid_metadata = all_data['valid']
        self.test_data, self.test_metadata = all_data['test']
        
        self.n_class=len(self.train_metadata['targets'])
        self.input_size=self.train_metadata['input_size']

def returnString(intArray):
    xxx={
            0:'a',
            1:'b',
            2:'c',
            3:'d',
            4:'e',
            5:'f',
            6:'g',
            7:'h',
            8:'i',
            9:'j',
            10:'k',
            11:'l',
            12:'m',
            13:'n',
            14:'o',
            15:'p',
            16:'q',
            17:'r',
            18:'s',
            19:'t',
            20:'u',
            21:'v',
            22:'w',
            23:'x',
            24:'y',
            25:'z',            
            }
    fString=""
    for x in intArray:
        fString+=xxx[x]   
    return fString        
    
def test(myObj,testArray):
    myObj.use(testArray)
    print(myObj.useOutput)
    print(myObj.useConfidence)
#test(myObj,[SamX.train_data[1][0]])
SamX=SampleData()

myObj=LinearChainCRF(n_epochs=100)
myObj.initialize(SamX.input_size,SamX.n_class)
myObj.train(SamX.train_data[1:2])



'''

inputX=[
        [1,2,3],
        [2,3,4],
        [3,4,5],        
        ]
inputY=[
        [3,4,15],
        [13,20,32],
        [12,23,4],        
        ]
target=[0,1,0]#not 0 indexed

myObj=LinearChainCRF()
myObj.initialize(3,2)




for i in range(10):
    #myObj.bias[0]+=0.0001
    #print("Bias:", myObj.bias)
    myObj.fprop(inputX)
    myObj.training_loss(target)  
    myObj.bprop(inputX,target)
    myObj.update()
myObj.use([inputX,inputY])
print(myObj.useOutput)
print(myObj.useConfidence)
#myObj.verify_gradients()

#print(myObj.alpha)
#print(myObj.beta)

'''

'''
print("weights:",myObj.weights)
print("bias:",myObj.bias)
print("Lateral W",myObj.lateral_weights)
print("unary table", myObj.target_unary_log_factors)
print("sumAlpha:",np.sum(myObj.alpha[-1:])," sumBeta:",np.sum(myObj.beta[:1])," diff:",np.sum(myObj.alpha[-1:])-np.sum(myObj.beta[:1]))
print("logAlpha:")
print(myObj.log_alpha)
print("logBeta:")
print(myObj.log_beta)
print("Marginal table:")
print(myObj.pykX)
print("Joint Marginal table:")
print(myObj.pykykP1X)
print("Grad Bias:")
print(myObj.grad_bias)
'''














