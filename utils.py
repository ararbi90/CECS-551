# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:27:33 2019

@author: Owner
"""

from __future__ import print_function
from scipy.misc import imread
from six.moves import cPickle as pickle
import numpy as np
import os
import platform
import csv

def csv2pickle(path_csv):
    training = {}
    public_test = {}
    private_test = {}
    pixels_train = []
    emotion_train = []
    pixels_pubtest = []
    emotion_pubtest = []
    pixels_pritest = []
    emotion_pritest = []
    with open(path_csv,'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader: 
            if(line[2] == "Training"):
                pixels_train += [int(s) for s in line[1].split()]
                emotion_train+= [int(line[0])]
            elif(line[2] == "PublicTest"):
                pixels_pubtest += [int(s) for s in line[1].split()]
                emotion_pubtest+= [int(line[0])]
            elif(line[2] == "PrivateTest"):
                pixels_pritest+= [int(s) for s in line[1].split()]
                emotion_pritest+= [int(line[0])]
        training["pixels"] = pixels_train
        training["emotion"]= emotion_train
        public_test["pixels"] = pixels_pubtest
        public_test["emotion"]= emotion_pubtest
        private_test["pixels"] = pixels_pritest
        private_test["emotion"]= emotion_pritest
        
    with open("train",'wb') as f:
        pickle.dump(training, f, pickle.HIGHEST_PROTOCOL)
    with open("publictest",'wb') as f:
        pickle.dump(public_test, f, pickle.HIGHEST_PROTOCOL)
    with open("privatetest",'wb') as f:
        pickle.dump(private_test, f, pickle.HIGHEST_PROTOCOL)
    
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_batch(filename):
  """ load single batch of cifar """
  with open('datasets/'+ filename, 'rb') as f:
    datadict = load_pickle(f)
    X = np.array(datadict['pixels'])
    Y = datadict['emotion']
    if filename == "train":
        X = X.reshape(28709, 1, 48, 48).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
    elif filename == "publictest":
        X = X.reshape(3589, 1, 48, 48).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
    elif filename == "privatetest":
        X = X.reshape(3589, 1, 48, 48).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
  return X, Y

def load_data():
  Xtr, Ytr = load_batch("train")
  Xpubte, Ypubte = load_batch("publictest")
  Xprite, Yprite = load_batch("privatetest")
  return Xtr, Ytr, Xpubte, Ypubte, Xprite, Yprite
