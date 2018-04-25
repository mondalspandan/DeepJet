# coding: utf-8

import sys, os
#import keras
#import tensorflow as tf

#from keras.losses import kullback_leibler_divergence, categorical_crossentropy
#from keras.models import load_model, Model
#from testing import testDescriptor
from argparse import ArgumentParser
#from keras import backend as K
#from Losses import * #needed!                                                                                                                
#from modelTools import fixLayersContaining,printLayerInfosAndWeights
                                                                                                                                  
#import numpy as np
#import matplotlib
#matplotlib.use('agg')
##import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, auc
#from root_numpy import array2root
#import pandas as pd
#import h5py

# Day tag for outputs
import datetime
now = datetime.datetime.now()
dayinfo = now.strftime("%b%d_%y")
# Options 
parser = ArgumentParser(description ='Script to run the training and evaluate it')
parser.add_argument("--train", action='store_true', default=False, help="Run training")
parser.add_argument("--eval", action='store_true', default=False, help="Run evaluation")
parser.add_argument("--re", action='store_true', default=False, help="Evaluation from stored df")
#parser.add_argument("-g", "--gpu", default=1, help="Run on gpu's (Need to be in deepjetLinux3_gpu env)", const=1)
parser.add_argument("-i", help="Training dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-n",  help="Training directory name will by default look like MMMDD_YYYY_train[your input]", default="test", metavar="PATH")
opts=parser.parse_args()
# Toggle training or eval
TrainBool = opts.train
EvalBool = opts.eval

# Detect GPU devices
#if opts.gpu: import setGPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Some used default settings
class MyClass:
    """A simple example class"""
    def __init__(self):
	self.gpu = ''
	self.gpufraction = ''
	self.modelMethod = ''
        self.inputDataCollection = ''
        self.outputDir = ''

sampleDatasets_pf_cpf_sv = ["db","pf","cpf","sv"]
sampleDatasets_cpf_sv = ["db","cpf","sv"]
sampleDatasets_sv = ["db","sv"]


# Setup inputs
if opts.i != None: trainDataCollection = opts.i
else: trainDataCollection = '/afs/cern.ch/work/a/anovak/public/Jan19_train_full/dataCollection.dc'
testDataCollection = trainDataCollection.replace("train","test")


#removedVars = [[],range(0,10), range(0,22),[0,1,2,3,4,5,6,7,8,9,10,13]]
removedVars = None

#Toggle to load model directly (True) or load weights (False) 
LoadModel = True

#select model and eval functions
from models.DeepJet_models_final import conv_model_final as trainingModel
#from training_base import training_base
from DeepJetCore.training.training_base import training_base
from eval_funcs import loadModel, makeRoc, _byteify, makeLossPlot, makeComparisonPlots


trainDir = opts.n
inputTrainDataCollection = trainDataCollection
inputTestDataCollection = testDataCollection
inputDataset = sampleDatasets_pf_cpf_sv

if TrainBool:
    args = MyClass()
    args.inputDataCollection = inputTrainDataCollection
    args.outputDir = trainDir
    #args.GPU = -1
    args.multi_gpu = os.system("nvidia-smi -L")+1
    print args.multi_gpu

    train=training_base(splittrainandtest=0.9,testrun=False, parser=args)
    #train=training_base(splittrainandtest=0.9,testrun=False, args=args)
    #if not train.modelSet():
    if True:
        #train.setModel(trainingModel,inputDataset,removedVars, multi_gpu=1 )
        train.setModel(trainingModel, 
			 #num_classes=5, #num_regclasses=5,
				datasets=inputDataset, multi_gpu=1)
    
        train.compileModel(learningrate=0.001,
                           loss=['categorical_crossentropy'],
                           	 metrics=['accuracy','binary_accuracy','MSE','MSLE'],
				loss_weights=[1.])
        """ 
	model,history,callbacks = train.trainModel(nepochs=1, 
                                                   batchsize=4000, 
                                                   stop_patience=1000, 
                                                   lr_factor=0.7, 
                                                   lr_patience=10, 
                                                   lr_epsilon=0.00000001, 
                                                   lr_cooldown=2, 
                                                   lr_minimum=0.00000001, 
                                                    maxqsize=100)
	
        train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')
        """
        model, history = train.trainModel(nepochs=5,
                                                   batchsize=5024,
                                                   stop_patience=1000,
                                                   lr_factor=0.7,
                                                   lr_patience=10,
                                                   lr_epsilon=0.00000001,
                                                   lr_cooldown=2,
                                                   lr_minimum=0.00000001,
                                                   maxqsize=100)

if EvalBool:
    evalModel = loadModel(trainDir,inputTrainDataCollection,trainingModel,LoadModel,inputDataset,removedVars)
    evalDir = trainDir.replace('train','out')
    
    from DeepJetCore.DataCollection import DataCollection
    testd=DataCollection()
    testd.readFromFile(inputTestDataCollection)

    if os.path.isdir(evalDir):
        raise Exception('output directory: %s must not exists yet' %evalDir)
    else:
        os.mkdir(evalDir)

    df = makeRoc(testd, evalModel, evalDir, opts.re)
    makeLossPlot(trainDir,evalDir)
    
