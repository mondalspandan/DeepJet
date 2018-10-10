import sys
import os
import keras
import tensorflow as tf

from keras.losses import kullback_leibler_divergence, categorical_crossentropy
from keras.models import load_model, Model
from argparse import ArgumentParser
from keras import backend as K
from Losses import * #needed!
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from root_numpy import array2root
import pandas as pd
import h5py
NBINS=40

#sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()

def loadModel(inputDir,trainData,model,LoadModel,sampleDatasets=None,removedVars=None):
    inputModel = '%s/KERAS_check_best_model.h5'%inputDir
  
    from DeepJetCore.DataCollection import DataCollection
    traind=DataCollection()
    traind.readFromFile(trainData)
    traind.dataclass.regressiontargetclasses = range(0,NBINS)
    print traind.getNRegressionTargets()

    if(LoadModel):
        evalModel = load_model(inputModel, custom_objects = global_loss_list)
        shapes=traind.getInputShapes()

    else:
        shapes=traind.getInputShapes()
        train_inputs = []
        for s in shapes:
            train_inputs.append(keras.layers.Input(shape=s))
        evalModel = model(train_inputs,traind.getNClassificationTargets(),traind.getNRegressionTargets(),sampleDatasets,removedVars)
        evalModel.load_weights(inputModel)

    return evalModel

def evaluate(testd, trainData, model, outputDir, storeInputs=False):
    	NENT = 1  # Can skip some events
    	filelist=[]
        i=0
        for s in testd.samples:
        #for s in testd.samples[0:1]:
            spath = testd.getSamplePath(s)
            filelist.append(spath)
            h5File = h5py.File(spath)
   	    f = h5File
            #features_val = [h5File['x%i_shape'%j][()] for j in range(0, h5File['x_listlength'][()][0])]
            features_val = [h5File['x%i'%j][()] for j in range(0, h5File['x_listlength'][()][0])]
            #features_val=testd.getAllFeatures()
            predict_test_i = model.predict(features_val)
            labels_val_i = h5File['y0'][()][::NENT,:]
            spectators_val_i = h5File['z0'][()][::NENT,0,:]
            if storeInputs: raw_features_val_i = h5File['z1'][()][::NENT,0,:]
            if i==0:
                predict_test = predict_test_i
                labels_val = labels_val_i
                spectators_val = spectators_val_i
                if storeInputs: raw_features_val = raw_features_val_i                                                    
            else:
                predict_test = np.concatenate((predict_test,predict_test_i))
                labels_val = np.concatenate((labels_val, labels_val_i))
                spectators_val = np.concatenate((spectators_val, spectators_val_i))
                if storeInputs: raw_features_val = np.concatenate((raw_features_val, raw_features_val_i))
            i+=1

        # Value
	#labels_val=testd.getAllLabels()[0][::NENT,:]
        #features_val=testd.getAllFeatures()[0][::NENT,0,:]
        #spectators_val = testd.getAllSpectators()[0][::NENT,0,:]
        #if storeInputs: raw_features_val = testd.getAllSpectators()[-1][::NENT,0,:]
        
	# Labels
	print testd.dataclass.branches
	feature_names = testd.dataclass.branches[1]
	spectator_names = testd.dataclass.branches[0]
        #truthnames = testd.getUsedTruth()
        
	from DeepJetCore.DataCollection import DataCollection                 
    	traind=DataCollection()
	traind.readFromFile(trainData)
	truthnames = traind.getUsedTruth()
	# Store features                                            
	print "Coulmns", spectator_names                   
        df = pd.DataFrame(spectators_val, columns = spectator_names)

	if storeInputs: 
		for i, tname in enumerate(feature_names):
			df[tname] = raw_features_val[:,i]

	# Add predictions
	print truthnames
	print predict_test.shape
	for i, tname in enumerate(truthnames):
		df['truth'+tname] = labels_val[:,i]
		#print "Mean 0th label predict predict of ", tname, np.mean(predict_test[:,0]), ", Stats:", np.sum(labels_val[:,i]), "/", len(labels_val[:,i])
		df['predict'+tname] = predict_test[:,i]

	print "Testing prediction:"
	print "Total: ", len(predict_test[:,0])
	for lab in truthnames:
		print lab, ":", sum(df['truth'+lab].values)

	df.to_pickle(outputDir+'/output.pkl')    #to save the dataframe, df to 123.pkl
	return df
	print "Finished storing dataframe"	

	    
