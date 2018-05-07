# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:49:59 2017

@author: PhantomV
"""

import ocr_letters_sequential as dl


#dl.load("")
dl.obtain("")

#array([ 4, 14,  6, 17,  0, 15,  7, 24]))
#array([14, 12, 15, 11,  4, 23]))
#array([22,  0,  1]))
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