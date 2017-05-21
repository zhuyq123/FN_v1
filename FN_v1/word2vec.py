# -*- coding: utf-8 -*-
"""
Created on Thu May 18 23:47:02 2017

@author: Administrator
"""

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(\
'GoogleNews-vectors-negative300-small.bin', binary=True)


if model.__contains__("happy"):
    w2v = model["happy"]
      
    