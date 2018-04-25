
# coding: utf-8

# In[ ]:

import sys
import os
import keras
#keras.backend.set_image_data_format('channels_last')
import tensorflow as tf

# In[3]:
from keras.losses import kullback_leibler_divergence, categorical_crossentropy
from keras.models import load_model, Model
from DeepJetCore.testing import testDescriptor
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
#from Losses import NBINS
NBINS=40

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

def loadModel(inputDir,trainData,model,LoadModel,sampleDatasets=None,removedVars=None):
    inputModel = '%s/KERAS_check_best_model.h5'%inputDir
    # inputModel = '%s/KERAS_model.h5'%inputDir
    inputWeights = '%s/KERAS_check_best_model_weights.h5' %inputDir
  
    from DeepJetCore.DataCollection import DataCollection
    traind=DataCollection()
    traind.readFromFile(trainData)
    traind.dataclass.regressiontargetclasses = range(0,NBINS)
    print traind.getNRegressionTargets()

    if(LoadModel):
        evalModel = load_model(inputModel, custom_objects = global_loss_list)
        #print evalModel.summary()
        shapes=traind.getInputShapes()

        #print traind.getInputShapes()

    else:
        shapes=traind.getInputShapes()
        train_inputs = []
        for s in shapes:
            train_inputs.append(keras.layers.Input(shape=s))
        evalModel = model(train_inputs,traind.getNClassificationTargets(),traind.getNRegressionTargets(),sampleDatasets,removedVars)
        evalModel.load_weights(inputWeights)

    return evalModel

def makeRoc(testd, model, outputDir, replot=False):
    ## # summarize history for loss for training and test sample
    ## plt.figure(1)
    ## plt.plot(callbacks.history.history['loss'])
    ## plt.plot(callbacks.history.history['val_loss'])
    ## plt.title('model loss')
    ## plt.ylabel('loss')
    ## plt.xlabel('epoch')
    ## plt.legend(['train', 'test'], loc='upper left')
    ## plt.savefig(self.outputDir+'learningcurve.pdf') 
    ## plt.close(1)

    ## plt.figure(2)
    ## plt.plot(callbacks.history.history['acc'])
    ## plt.plot(callbacks.history.history['val_acc'])
    ## plt.title('model accuracy')
    ## plt.ylabel('acc')
    ## plt.xlabel('epoch')
    ## plt.legend(['train', 'test'], loc='upper left')
    ## plt.savefig(self.outputDir+'accuracycurve.pdf')
    ## plt.close(2)

    print 'in makeRoc()'
    
    # let's use all entries
    NENT = 1

    if replot:
	df = pd.read_pickle(outputDir.replace("out","train")+'/testresults.pkl')    
    else:
        #print testd.nsamples
        filelist=[]
        i=0
        for s in testd.samples:
            spath = testd.getSamplePath(s)
            print spath
            filelist.append(spath)
            h5File = h5py.File(spath)
   	    f = h5File
	    print "Keys: %s" % f.keys()
            print f['x_listlength']
            #features_val = [h5File['x%i_shape'%j][()] for j in range(0, h5File['x_listlength'][()][0])]
            features_val = [h5File['x%i'%j][()] for j in range(0, h5File['x_listlength'][()][0])]
            # print features_val
            predict_test_i = model.predict(features_val)
            if i==0:
                predict_test = predict_test_i
            else:
                predict_test = np.concatenate((predict_test,predict_test_i))
            i+=1
	    print "TEST SHAPE"
            print predict_test.shape

    #features_val = [fval[::NENT] for fval in testd.getAllFeatures()]
        labels_val=testd.getAllLabels()[0][::NENT,:]
    #weights_val=testd.getAllWeights()[0][::NENT]
    #spectators_val = testd.getAllSpectators()[0][::NENT,0,:]

    #print features_val[0].shape
        truthnames = testd.getUsedTruth()
                                                                                        
        spectators_val = testd.getAllSpectators()[0][::NENT,0,:]
        df = pd.DataFrame(spectators_val)
        df.columns = ['fj_pt',
                  'fj_eta',
                  'fj_sdmass',
                  'fj_n_sdsubjets',
                  'fj_doubleb',
                  'fj_tau21',
                  'fj_tau32',
                  'npv',
                  'npfcands',
                  'ntracks',
                  'nsv']

    #print(df.iloc[:10])

        
    #predict_test = model.predict(features_val)
    #print predict_test 

    #predict_tf = tf.convert_to_tensor(predict_test, np.float32)
    #OH_tf = tf.convert_to_tensor(OH, np.float32)
    #print('loss_kldiv',loss_kldiv(OH_tf,predict_tf).eval())
	print labels_val[0]
	print predict_test[0]

	for i, tname in enumerate(truthnames):
		if len(truthnames) == 2: # to be fixed
			if i == 0: 
				df['fj_isH'] = labels_val[:,1]
			if i == 1:
				df['fj_deepdoublec'] = predict_test[:,1]
		else:
				
			df['truth'+tname] = labels_val[:,i]
			df['predict'+tname] = predict_test[:,i]
        #df['fj_isH'] = labels_val[:,1]
        #df['fj_deepdoublec'] = predict_test[:,1]

        df = df[(df.fj_sdmass > 40) & (df.fj_sdmass < 200) & (df.fj_pt > 300) &  (df.fj_pt < 2500)]

	df.to_pickle(outputDir+'/testresults.pkl')    #to save the dataframe, df to 123.pkl

    #print(df.iloc[:10])

    fpr, tpr, threshold = roc_curve(df['fj_isH'],df['fj_deepdoublec'])
    dfpr, dtpr, threshold1 = roc_curve(df['fj_isH'],df['fj_doubleb'])

    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]


    deepdoublebcuts = {}
    for wp in [0.01, 0.05, 0.1, 0.25, 0.5]: # % mistag rate
        idx, val = find_nearest(fpr, wp)
        deepdoublebcuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate
        print('deep double-b > %f coresponds to %f%% QCD mistag rate'%(deepdoublebcuts[str(wp)] ,100*val))

    auc1 = auc(fpr, tpr)
    auc2 = auc(dfpr, dtpr)

    plt.figure()       
    plt.plot(tpr,fpr,label='deep double-c, auc = %.1f%%'%(auc1*100))
    plt.plot(dtpr,dfpr,label='BDT double-b, auc = %.1f%%'%(auc2*100))
    plt.semilogy()
    plt.xlabel("H(cc) efficiency")
    plt.ylabel("QCD mistag rate")
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend()
    plt.savefig(outputDir+"/test.pdf")
    
    plt.figure()
    bins = np.linspace(-1,1,70)
    plt.hist(df['fj_doubleb'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_doubleb'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(cc)')
    plt.xlabel("BDT double-b")
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"/doubleb.pdf")

    plt.figure()
    bins = np.linspace(0,1,70)
    plt.hist(df['fj_deepdoublec'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_deepdoublec'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(cc)')
    plt.xlabel("deep double-b")
    #plt.ylim(0.00001,1)
    #plt.semilogy()
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"/deepdoublec.pdf")
    
    plt.figure()
    bins = np.linspace(0,2000,70)
    plt.hist(df['fj_pt'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_pt'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(cc)')
    plt.xlabel(r'$p_{\mathrm{T}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"/pt.pdf")
    
    plt.figure()
    bins = np.linspace(40,200,41)
    plt.hist(df['fj_sdmass'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_sdmass'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(cc)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"/msd.pdf")
    
    plt.figure()
    bins = np.linspace(40,200,41)
    df_passdoubleb = df[df.fj_doubleb > 0.9]
    plt.hist(df_passdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdoubleb['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df_passdoubleb['fj_sdmass'], bins=bins, weights = df_passdoubleb['fj_isH'],alpha=0.5,normed=True,label='H(cc)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"/msd_passdoubleb.pdf")

    plt.figure()
    bins = np.linspace(40,200,41)
    for wp, deepdoublebcut in reversed(sorted(deepdoublebcuts.iteritems())):
        df_passdeepdoubleb = df[df.fj_deepdoublec > deepdoublebcut]
        plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdeepdoubleb['fj_isH'],normed=True,histtype='step',label='QCD %i%% mis-tag'%(float(wp)*100.))
        #plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = df_passdeepdoubleb['fj_isH'],alpha=0.5,normed=True,label='H(cc) %s'%wp)
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper right')
    plt.savefig(outputDir+"/msd_passdeepdoublec.pdf")
    
    plt.figure()
    bins = np.linspace(40,200,41)
    plt.hist(df['fj_sdmass'], bins=bins, weights = (1-df['fj_deepdoublec'])*(1-df['fj_isH']),alpha=0.5,normed=True,label='p(QCD|QCD)')
    plt.hist(df['fj_sdmass'], bins=bins, weights = df['fj_deepdoublec']*(1-df['fj_isH']),alpha=0.5,normed=True,label='p(H(cc)|QCD)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"/msd_weightdeepdoublec_QCD.pdf")

    plt.figure()
    bins = np.linspace(40,200,41)
    plt.hist(df['fj_sdmass'], bins=bins, weights = (1-df['fj_deepdoublec'])*(df['fj_isH']),alpha=0.5,normed=True,label='p(QCD|H(cc))')
    plt.hist(df['fj_sdmass'], bins=bins, weights = df['fj_deepdoublec']*(df['fj_isH']),alpha=0.5,normed=True,label='p(H(cc)|H(cc))')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"/msd_weightdeepdoublec_H.pdf")


    def kl(p, q):
        p = np.asarray(p, dtype=np.float32)
        q = np.asarray(q, dtype=np.float32)
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    hist_anti_q, bin_edges = np.histogram(df['fj_sdmass'],bins =40, range = (40, 200), weights = (1-df['fj_deepdoublec'])*(1-df['fj_isH']),normed=True)
    hist_fill_q, bin_edges = np.histogram(df['fj_sdmass'],bins =40, range = (40, 200), weights = (df['fj_deepdoublec'])*(1-df['fj_isH']),normed=True)
    hist_anti_h, bin_edges = np.histogram(df['fj_sdmass'],bins =40, range = (40, 200), weights = (1-df['fj_deepdoublec'])*(df['fj_isH']),normed=True)
    hist_fill_h, bin_edges = np.histogram(df['fj_sdmass'],bins =40, range = (40, 200), weights = (df['fj_deepdoublec'])*(df['fj_isH']),normed=True)
    hist_anti_q=hist_anti_q*4.
    hist_fill_q=hist_fill_q*4.
    hist_anti_h=hist_anti_h*4.
    hist_fill_h=hist_fill_h*4.
    print('hist_fill_q sum', np.sum(hist_fill_q))
    print('hist_anti_q sum', np.sum(hist_anti_q))
    print('hist_fill_h sum', np.sum(hist_fill_h))
    print('hist_anti_h sum', np.sum(hist_anti_h))
    plt.figure()
    plt.bar(bin_edges[:-1], hist_anti_q, width = 4,alpha=0.5,label='p(QCD|QCD)')
    plt.bar(bin_edges[:-1], hist_fill_q, width = 4,alpha=0.5,label='p(H(cc)|QCD))')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"msd_kldiv_q.pdf")
    plt.figure()
    plt.bar(bin_edges[:-1], hist_anti_h, width = 4,alpha=0.5,label='p(QCD|H(cc))')
    plt.bar(bin_edges[:-1], hist_fill_h, width = 4,alpha=0.5,label='p(H(cc)|H(cc))')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"msd_kldiv_h.pdf")
    print('kl_q',kl(hist_fill_q, hist_anti_q))
    print('hist_fill_q',hist_fill_q)
    print('hist_anti_q',hist_anti_q)
    print('kl_h',kl(hist_fill_h, hist_anti_h))
    print('hist_fill_h',hist_fill_h)
    print('hist_anti_h',hist_anti_h)
    print('kl_q+kl_h',kl(hist_fill_q, hist_anti_q)+kl(hist_fill_h, hist_anti_h))

    return df

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data

def makeLossPlot(inputDir, outputDir):
    import json
    inputLogs = '%s/full_info.log'%inputDir
    f = open(inputLogs)
    myListOfDicts = json.load(f, object_hook=_byteify)
    myDictOfLists = {}
    for key, val in myListOfDicts[0].iteritems():
        myDictOfLists[key] = []
    for i, myDict in enumerate(myListOfDicts):
        for key, val in myDict.iteritems():
            myDictOfLists[key].append(myDict[key])
    val_loss = np.asarray(myDictOfLists['val_loss'])
    loss = np.asarray(myDictOfLists['loss'])
    plt.figure()
    plt.plot(val_loss, label='validation')
    plt.plot(loss, label='train')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("%s/loss.pdf"%outputDir)
    
def makeComparisonPlots(testds, models,names, outputDir, pkls=None, flips=None, xlabel="H(cc) efficiency", fout="ROCcomparison.pdf"):

# let's use all entries
    NENT = 1
    if pkls == None:   
	filelist=[]
        predictions = []
        for model, testd in zip(models, testds):
            first = True
            for s in testd.samples:
                spath = testd.getSamplePath(s)
                filelist.append(spath)
                h5File = h5py.File(spath)
                features_val = [h5File['x%i'%j][()] for j in range(0, h5File['x_listlength'][()][0])]
                predict_test_i = model.predict(features_val)
                if first:
                    predict_test = predict_test_i
                else:
                    predict_test = np.concatenate((predict_test,predict_test_i))
                first= False
            predictions.append(predict_test)

        labels_val=testd.getAllLabels()[0][::NENT,:]
                                                                                        
        spectators_val = testd.getAllSpectators()[0][::NENT,0,:]
    
        dfs = []
        for i in range(len(models)):
            df = pd.DataFrame(spectators_val)
            df.columns = ['fj_pt',
                      'fj_eta',
                      'fj_sdmass',
                      'fj_n_sdsubjets',
                      'fj_doubleb',
                      'fj_tau21',
                      'fj_tau32',
                      'npv',
                      'npfcands',
                      'ntracks',
                      'nsv']

            df['fj_isH'] = labels_val[:,1]
            df['fj_deepdoublec'] = predictions[i][:,1]
        
            df = df[(df.fj_sdmass > 40) & (df.fj_sdmass < 200) & (df.fj_pt > 300) &  (df.fj_pt < 2500)]

            dfs.append(df)
    else:
	dfs = []
	for pkl in pkls:
	    df = pd.read_pickle(pkl)	    
	    dfs.append(df)

    fprs = []
    tprs = []
    thresholds = []

    for i in range(len(names)):
        fpr, tpr, threshold = roc_curve(dfs[i]['fj_isH'],dfs[i]['fj_deepdoublec'])
        fprs.append(fpr)
        tprs.append(tpr)
        thresholds.append(threshold)

    dfpr, dtpr, threshold1 = roc_curve(dfs[0]['fj_isH'],dfs[0]['fj_doubleb'])

    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]

    
    
    deepdoublebcuts = {}
    workingPoints = [0.01, 0.05, 0.1, 0.25, 0.5] # % mistag rate
    for wp in workingPoints:
        deepdoublebcuts[str(wp)]=[]
    for i in range(len(names)):
        for wp in workingPoints: 
            idx, val = find_nearest(fpr, wp)
            deepdoublebcuts[str(wp)].append(threshold[idx]) # threshold for deep double-b corresponding to ~1% mistag rate
            print('deep double-b > %f coresponds to %f%% QCD mistag rate'%(deepdoublebcuts[str(wp)][i] ,100*val))

    aucs = []
    for i in range(len(names)):
	if flips!= None:
		if not flips[i]:
	        	aucs.append(auc(fprs[i],tprs[i]))
		else:
		        aucs.append(auc(1.-fprs[i],1.-tprs[i]))
	else:
		aucs.append(auc(fprs[i],tprs[i]))

    if flips == None: # nothing flipped
   	aucBDT = auc(dfpr, dtpr)
    else:
   	aucBDT = auc(1.-dfpr, 1.-dtpr)	

    plt.figure()       
    colors = ['firebrick', 'steelblue', 'green']
    for i in range(len(names)):
	c = colors[i]
	if flips!= None:
		if not flips[i]:
		        plt.plot(tprs[i],fprs[i], color=c, label='%s, auc = %.1f%%'% (names[i],(aucs[i]*100)))
		else:
	        	plt.plot(1.-tprs[i],1.-fprs[i], color = c, label='%s, auc = %.1f%%'% (names[i],(aucs[i]*100)))
	else:	
		plt.plot(tprs[i],fprs[i], color=c, label='%s, auc = %.1f%%'% (names[i],(aucs[i]*100)))

    if flips == None: # nothing flipped
	plt.plot(dtpr,dfpr, color='darkorange', label='BDT double-b, auc = %.1f%%'%(aucBDT*100))
    else:
	plt.plot(1.-dtpr,1.-dfpr, color='darkorange', label='BDT double-b, auc = %.1f%%'%(aucBDT*100))
    plt.semilogy()
    plt.xlabel(xlabel)
    if flips == None: # nothing flipped
        plt.ylabel("QCD mistag rate")
    else:
	plt.ylabel("H(bb) mistag rate")
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend()
    plt.savefig(fout, dpi=900)


    for wp in workingPoints:
	continue
        plt.figure()
        bins = np.linspace(40,200,41)
        for i in range(len(names)):
            deepdoublebcut = deepdoublebcuts[str(wp)][i]
            df_passdeepdoubleb = dfs[i][dfs[i].fj_deepdoublec > deepdoublebcut]
            plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdeepdoubleb['fj_isH'],alpha=0.5,normed=True,label='QCD %i%% mis-tag for %s'%(float(wp)*100.,names[i]))
        #plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = df_passdeepdoubleb['fj_isH'],alpha=0.5,normed=True,label='H(cc) %s'%wp)
        plt.xlabel(r'$m_{\mathrm{SD}}$')
        plt.legend(loc='upper right')
        plt.savefig(outputDir+"msd_passdeepdoubleb%s.pdf"%(str(int(wp*100))))
