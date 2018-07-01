
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
from argparse import ArgumentParser
from keras import backend as K
from Losses import loss_NLL, loss_meansquared, loss_kldiv, global_loss_list
from Layers import global_layers_list
from Metrics import global_metrics_list
custom_objects_list = {}
custom_objects_list.update(global_loss_list)
custom_objects_list.update(global_layers_list)
custom_objects_list.update(global_metrics_list)
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from root_numpy import array2root
import pandas as pd
import h5py
from Losses import NBINS

sess = tf.InteractiveSession()

def loadModel(inputDir,trainData,model,LoadModel,forceNClasses = False, NClasses = 2, sampleDatasets=None,removedVars=None):
    inputModel = '%s/KERAS_check_best_model.h5'%inputDir
  
    from DeepJetCore.DataCollection import DataCollection

    traind=DataCollection()
    traind.readFromFile(trainData)
    #traind.dataclass.regressiontargetclasses = range(0,NBINS)
    #print traind.getNRegressionTargets()

    if(LoadModel):
        evalModel = load_model(inputModel, custom_objects = custom_objects_list)
        print evalModel.summary()
    else:
        if forceNClasses:
            nClasses = NClasses
        else:
            nClasses = traind.getNClassificationTargets()
        shapes=traind.getInputShapes()
        train_inputs = []
        for s in shapes:
            train_inputs.append(keras.layers.Input(shape=s))
        evalModel = model(train_inputs,nClasses,traind.getNRegressionTargets(),sampleDatasets,removedVars)
        evalModel.load_weights(inputModel)

    return evalModel


def makePlots(testd, model,outputDir,signals = [1], sigNames = ["Hbb"], backgrounds = [0],backNames = ["qcd"]):
    print 'in makeRoc()'


    NENT = 1
    
    print testd.nsamples
    filelist=[]
    i=0
    for s in testd.samples[0:10]:
    #for s in testd.samples:
        spath = testd.getSamplePath(s)
        print spath
        filelist.append(spath)
        h5File = h5py.File(spath)
        features_val = [h5File['x%i'%j][()] for j in range(0, h5File['x_listlength'][()][0])]
        predict_test_i = model.predict(features_val)
        labels_val_i = h5File['y0'][()][::NENT,:]
        spectators_val_i = h5File['z0'][()][::NENT,0,:]
        if i==0:
            predict_test = predict_test_i
            labels_val = labels_val_i
            spectators_val = spectators_val_i
        else:
#            if type(predict_test) is list:
#                predict_test = [np.concatenate((p_t,p_t_i)) for p_t, p_t_i in zip(predict_test,predict_test_i)]
#            else:
            predict_test = np.concatenate((predict_test,predict_test_i))
            labels_val = np.concatenate((labels_val, labels_val_i))
            spectators_val = np.concatenate((spectators_val, spectators_val_i))
        i+=1
     
    #objects =testd.getAllLabelsFeaturesSpectators()
    #labels_val=objects[1][0][::NENT,:]
    #spectators_val = objects[2][0][::NENT,0,:]
    
    # get all files
    #labels_val=testd.getAllLabels()[0][::NENT,:]
    #spectators_val = testd.getAllSpectators()[0][::NENT,0,:]

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

    print(df.iloc[:10])

    df['fj_deepdoubleb'] = predict_test[:,1]
    
    for i in range(len(signals)):
        df['fj_isSig_%s'%sigNames[i]] = labels_val[:,signals[i]]
        if i == 0:
            df['fj_deepdoublebSig'] = predict_test[:,1]
            df['fj_isSignal'] = df['fj_isSig_%s'%sigNames[i]]
        else:
            df['fj_isSignal'] = np.add(df['fj_isSignal'],df['fj_isSig_%s'%sigNames[i]])
        

    for i in range(len(backgrounds)):
        df['fj_isBack_%s'%backNames[i]] = labels_val[:,backgrounds[i]]
        if i ==0:
            df['fj_isBackground'] =df['fj_isBack_%s'%backNames[i]]
        else:
            df['fj_isBackground'] =np.add(df['fj_isBackground'],df['fj_isBack_%s'%backNames[i]])
    dfOrig = df
    ptBounds = [(300,500),(500,800),(800,2500),(300,2500)]
    outputOrig= outputDir
    for ptTuple in ptBounds:
        lowBound = ptTuple[0]
        highBound = ptTuple[1]
        outputDir = outputOrig + '/PT_%d_%d/'%(lowBound,highBound)
        for i in range(len(signals)):
            allBackName = ""
            first = True
            for j in range(len(backgrounds)):
                if first:
                    allBackName = backNames[j]
                    first = False
                else:
                    allBackName = allBackName + "_" + backNames[j]
                dfCur = dfOrig[(dfOrig.fj_sdmass > 40) & (dfOrig.fj_sdmass < 200) & (dfOrig.fj_pt > lowBound) &  (dfOrig.fj_pt < highBound)]
                print dfCur
                dfCur = dfCur.query('fj_isSig_%s ==1 | fj_isBack_%s == 1'%(sigNames[i],backNames[j]))
                print dfCur
                if not os.path.isdir(outputDir):
                    os.mkdir(outputDir)
                makeRocPlots(dfCur['fj_isSig_%s'%sigNames[i]],dfCur['fj_deepdoublebSig'],dfCur['fj_doubleb'],sigNames[i], backNames[j],outputDir + 'rocCurve_sig%s_back%s_pt_%d_%d.pdf'%(sigNames[i],backNames[j],lowBound,highBound))
            dfCur = dfOrig[(dfOrig.fj_sdmass > 40) & (dfOrig.fj_sdmass < 200) & (dfOrig.fj_pt > lowBound) &  (dfOrig.fj_pt < highBound)]
            print dfCur
            dfCur = dfCur.query('fj_isSig_%s ==1 | fj_isBackground == 1'%(sigNames[i]))
            print dfCur
            if not os.path.isdir(outputDir):
                os.mkdir(outputDir)
            makeRocPlots(dfCur['fj_isSig_%s'%sigNames[i]],dfCur['fj_deepdoublebSig'],dfCur['fj_doubleb'],sigNames[i], allBackName,outputDir + 'rocCurve_sig%s_back%s_pt_%d_%d.pdf'%(sigNames[i],allBackName,lowBound,highBound))


        df = dfOrig[(dfOrig.fj_sdmass > 40) & (dfOrig.fj_sdmass < 200) & (dfOrig.fj_pt > lowBound) &  (dfOrig.fj_pt < highBound)] # & (df.fj_isSignal==1 or df.fj_isBackground==1)]

        sigLabel = ""
        for s in sigNames:
            makeVarPlot(np.linspace(0,2000,70),df['fj_pt'],df['fj_isSignal'],'H(bb)',df['fj_isBackground'], 'QCD', r'$p_{\mathrm{T}}$', outputDir+"pt.pdf")
            sigLabel = sigLabel + ";"+ s

        backLabel = ""
        for s in backNames:
            backLabel = backLabel + ";" + s

        deepdoublebcuts = makeRocPlots(df['fj_isSignal'],df['fj_deepdoublebSig'],df['fj_doubleb'],sigLabel, backLabel, outputDir+"rocCurve.pdf")


        makeVarPlot(np.linspace(-1,1,70),df['fj_doubleb'],df['fj_isSignal'],'H(bb)',df['fj_isBackground'], 'QCD', "BDT double-b", outputDir+"doubleb.pdf")
        makeVarPlot(np.linspace(0,1,70),df['fj_deepdoublebSig'],df['fj_isSignal'],'H(bb)',df['fj_isBackground'], 'QCD', "deep double-b", outputDir+"deepdoubleb.pdf")
        makeVarPlot(np.linspace(0,2000,70),df['fj_pt'],df['fj_isSignal'],'H(bb)',df['fj_isBackground'], 'QCD', r'$p_{\mathrm{T}}$', outputDir+"pt.pdf")
        makeVarPlot(np.linspace(40,200,41),df['fj_sdmass'],df['fj_isSignal'],'H(bb)',df['fj_isBackground'], 'QCD', r'$m_{\mathrm{SD}}$', outputDir+"msd.pdf")
    
        plt.figure()
        bins = np.linspace(40,200,41)
        df_passdoubleb = df[df.fj_doubleb > 0.9]
        plt.hist(df_passdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdoubleb['fj_isSignal'],alpha=0.5,normed=True,label='QCD')
        plt.hist(df_passdoubleb['fj_sdmass'], bins=bins, weights = df_passdoubleb['fj_isSignal'],alpha=0.5,normed=True,label='H(bb)')
        plt.xlabel(r'$m_{\mathrm{SD}}$')
        plt.legend(loc='upper left')
        plt.savefig(outputDir+"msd_passdoubleb.pdf")

        plt.figure()
        bins = np.linspace(40,200,41)
        for wp, deepdoublebcut in reversed(sorted(deepdoublebcuts.iteritems())):
            df_passdeepdoubleb = df[df.fj_deepdoublebSig > deepdoublebcut]
            plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdeepdoubleb['fj_isSignal'],normed=True,histtype='step',label='QCD %i%% mis-tag'%(float(wp)*100.))
            #plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = df_passdeepdoubleb['fj_isSignal'],alpha=0.5,normed=True,label='H(bb) %s'%wp)
        plt.xlabel(r'$m_{\mathrm{SD}}$')
        plt.legend(loc='upper right')
        plt.savefig(outputDir+"msd_passdeepdoubleb.pdf")
    
        plt.figure()
        bins = np.linspace(40,200,41)
        plt.hist(df['fj_sdmass'], bins=bins, weights = (1-df['fj_deepdoublebSig'])*(1-df['fj_isSignal']),alpha=0.5,normed=True,label='p(QCD|QCD)')
        plt.hist(df['fj_sdmass'], bins=bins, weights = df['fj_deepdoublebSig']*(1-df['fj_isSignal']),alpha=0.5,normed=True,label='p(H(bb)|QCD)')
        plt.xlabel(r'$m_{\mathrm{SD}}$')
        plt.legend(loc='upper left')
        plt.savefig(outputDir+"msd_weightdeepdoubleb_QCD.pdf")

        plt.figure()
        bins = np.linspace(40,200,41)
        plt.hist(df['fj_sdmass'], bins=bins, weights = (1-df['fj_deepdoublebSig'])*(df['fj_isSignal']),alpha=0.5,normed=True,label='p(QCD|H(bb))')
        plt.hist(df['fj_sdmass'], bins=bins, weights = df['fj_deepdoublebSig']*(df['fj_isSignal']),alpha=0.5,normed=True,label='p(H(bb)|H(bb))')
        plt.xlabel(r'$m_{\mathrm{SD}}$')
        plt.legend(loc='upper left')
        plt.savefig(outputDir+"msd_weightdeepdoubleb_H.pdf")


        def kl(p, q):
            p = np.asarray(p, dtype=np.float32)
            q = np.asarray(q, dtype=np.float32)
            return np.sum(np.where(p != 0, p * np.log(p / q), 0))
            
        hist_anti_q, bin_edges = np.histogram(df['fj_sdmass'],bins =40, range = (40, 200), weights = (1-df['fj_deepdoublebSig'])*(1-df['fj_isSignal']),normed=True)
        hist_fill_q, bin_edges = np.histogram(df['fj_sdmass'],bins =40, range = (40, 200), weights = (df['fj_deepdoublebSig'])*(1-df['fj_isSignal']),normed=True)
        hist_anti_h, bin_edges = np.histogram(df['fj_sdmass'],bins =40, range = (40, 200), weights = (1-df['fj_deepdoublebSig'])*(df['fj_isSignal']),normed=True)
        hist_fill_h, bin_edges = np.histogram(df['fj_sdmass'],bins =40, range = (40, 200), weights = (df['fj_deepdoublebSig'])*(df['fj_isSignal']),normed=True)
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
        plt.bar(bin_edges[:-1], hist_fill_q, width = 4,alpha=0.5,label='p(H(bb)|QCD))')
        plt.legend(loc='upper left')
        plt.savefig(outputDir+"msd_kldiv_q.pdf")
        plt.figure()
        plt.bar(bin_edges[:-1], hist_anti_h, width = 4,alpha=0.5,label='p(QCD|H(bb))')
        plt.bar(bin_edges[:-1], hist_fill_h, width = 4,alpha=0.5,label='p(H(bb)|H(bb))')
        plt.legend(loc='upper left')
        plt.savefig(outputDir+"msd_kldiv_h.pdf")
        print('kl_q',kl(hist_fill_q, hist_anti_q))
        print('hist_fill_q',hist_fill_q)
        print('hist_anti_q',hist_anti_q)
        print('kl_h',kl(hist_fill_h, hist_anti_h))
        print('hist_fill_h',hist_fill_h)
        print('hist_anti_h',hist_anti_h)
        print('kl_q+kl_h',kl(hist_fill_q, hist_anti_q)+kl(hist_fill_h, hist_anti_h))

    return dfOrig, features_val

def makeRocPlots(signal,deepdoubleb,doubleb, signalLabel, backgroundLabel, fileout):
    fpr, tpr, threshold = roc_curve(signal,deepdoubleb)
    dfpr, dtpr, dthreshold = roc_curve(signal,doubleb)
    
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
    plt.plot(tpr,fpr,label='deep double-b, auc = %.1f%%'%(auc1*100))
    plt.plot(dtpr,dfpr,label='BDT double-b, auc = %.1f%%'%(auc2*100))
    plt.semilogy()
    plt.xlabel('Signal:%s efficiency'%signalLabel)
    plt.ylabel('Background: %s  mistag rate' %backgroundLabel)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend()
    plt.savefig(fileout)

    return deepdoublebcuts

def makeVarPlot(bins,variable,signal,sigLabel,background, backLabel, plotLabel, fileout,alpha = 0.5, normed = True, legendLoc = 'upper left'):
    plt.figure()
    plt.hist(variable, bins=bins, weights = background,alpha=alpha,normed=normed,label=backLabel)
    plt.hist(variable, bins=bins, weights = signal,alpha=alpha,normed=normed,label=sigLabel)
    plt.xlabel(plotLabel)
    plt.legend(loc=legendLoc)
    plt.savefig(fileout)

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
    
def makeComparisonPlots(testds, models,names, outputDir):


# let's use all entries
    NENT = 1
    
    filelist=[]

    predictions = []
    dfs = []
    for i in range(len(models)):
        model = models[i]
        testd = testds[i]
        first = True
        for s in testd.samples:
            spath = testd.getSamplePath(s)
            print spath
            filelist.append(spath)
            h5File = h5py.File(spath)
            features_val = [h5File['x%i'%j][()] for j in range(0, h5File['x_listlength'][()][0])]
            predict_test_i = model.predict(features_val)
            if first:
                predict_test = predict_test_i
                first=False
            else:

                if type(predict_test) is list:
                    predict_test = [np.concatenate((p_t,p_t_i)) for p_t, p_t_i in zip(predict_test,predict_test_i)]
                else:
                    predict_test = np.concatenate((predict_test,predict_test_i))
            
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

        df['fj_isSignal'] = labels_val[:,1]
        df['fj_deepdoubleb'] = predict_test[:,1]
        
        df = df[(df.fj_sdmass > 40) & (df.fj_sdmass < 200) & (df.fj_pt > 300) &  (df.fj_pt < 2500)]

        dfs.append(df)

    fprs = []
    tprs = []
    thresholds = []

    for i in range(len(models)):
        fpr, tpr, threshold = roc_curve(dfs[i]['fj_isSignal'],dfs[i]['fj_deepdoubleb'])
        fprs.append(fpr)
        tprs.append(tpr)
        thresholds.append(threshold)

    dfpr, dtpr, threshold1 = roc_curve(dfs[0]['fj_isSignal'],dfs[0]['fj_doubleb'])

    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]
        
    deepdoublebcuts = {}
    workingPoints = [0.01, 0.05, 0.1, 0.25, 0.5] # % mistag rate
    for wp in workingPoints:
        deepdoublebcuts[str(wp)]=[]
    for i in range(len(models)):
        for wp in workingPoints: 
            idx, val = find_nearest(fpr, wp)
            deepdoublebcuts[str(wp)].append(threshold[idx]) # threshold for deep double-b corresponding to ~1% mistag rate
            print('deep double-b > %f coresponds to %f%% QCD mistag rate'%(deepdoublebcuts[str(wp)][i] ,100*val))

    aucs = []
    for i in range(len(models)):
        aucs.append(auc(fprs[i],tprs[i]))
    
    aucBDT = auc(dfpr, dtpr)

    plt.figure()       
    for i in range(len(models)):
        plt.plot(tprs[i],fprs[i],label='deep double-b %s, auc = %.1f%%'% (names[i],(aucs[i]*100)))

    plt.plot(dtpr,dfpr,label='BDT double-b, auc = %.1f%%'%(aucBDT*100))
    plt.semilogy()
    plt.xlabel("H(bb) efficiency")
    plt.ylabel("QCD mistag rate")
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend()
    plt.savefig(outputDir+"ROCcomparison.pdf")


    for wp in workingPoints:
        plt.figure()
        bins = np.linspace(40,200,41)
        for i in range(len(models)):
            deepdoublebcut = deepdoublebcuts[str(wp)][i]
            df_passdeepdoubleb = dfs[i][dfs[i].fj_deepdoubleb > deepdoublebcut]
            plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdeepdoubleb['fj_isSignal'],alpha=0.5,normed=True,label='QCD %i%% mis-tag for %s'%(float(wp)*100.,names[i]))

        plt.xlabel(r'$m_{\mathrm{SD}}$')
        plt.legend(loc='upper right')
        plt.savefig(outputDir+"msd_passdeepdoubleb%s.pdf"%(str(int(wp*100))))

    for wp in workingPoints:
        plt.figure()
        bins = np.linspace(40,200,41)
        for i in range(len(models)):
            deepdoublebcut = deepdoublebcuts[str(wp)][i]
            df_passdeepdoubleb = dfs[i][dfs[i].fj_deepdoubleb > deepdoublebcut]
            plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdeepdoubleb['fj_isSignal'],normed=True,histtype='step',label='QCD %i%% mis-tag for %s'%(float(wp)*100.,names[i]))

        plt.xlabel(r'$m_{\mathrm{SD}}$')
        plt.legend(loc='upper right')
        plt.savefig(outputDir+"msd_passdeepdoubleb_step%s.pdf"%(str(int(wp*100))))
