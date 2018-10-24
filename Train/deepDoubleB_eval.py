import sys, os
from argparse import ArgumentParser
import setGPU
os.environ['CUDA_VISIBLE_DEVICES']=''
# Options 
parser = ArgumentParser(description ='Script to run the training and evaluate it')
parser.add_argument("-i", help="Training dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-t", help="Testing dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-d",  help="Training output dir", default=None, metavar="PATH")
parser.add_argument("-o",  help="Eval output dir", default=None, metavar="PATH")
opts=parser.parse_args()

sampleDatasets_cpf_sv = ["db","cpf","sv"]

#select model and eval functions

from models import model_DeepDoubleXReference as trainingModel

from DeepJetCore.training.training_base import training_base
from eval_funcs import loadModel, makePlots, _byteify, makeLossPlot, makeComparisonPlots
#from eval_functions import loadModel, evaluate
#from plots_from_df import make_plots
from Losses import loss_NLL, loss_meansquared, loss_kldiv, global_loss_list
from Layers import global_layers_list
from Metrics import global_metrics_list
custom_objects_list = {}
custom_objects_list.update(global_loss_list)
custom_objects_list.update(global_layers_list)
custom_objects_list.update(global_metrics_list)

inputDataset = sampleDatasets_cpf_sv
trainDir = opts.d
inputTrainDataCollection = opts.i
inputTestDataCollection = opts.t
LoadModel = False
removedVars = None
forceNClasses = False
signals = [1]
sigNames = ['Hbb']
backgrounds = [0]
backNames = ['QCD']
NClasses = len(signals) + len(backgrounds)

if True:
    evalModel = loadModel(trainDir,inputTrainDataCollection,trainingModel,LoadModel,forceNClasses, NClasses,inputDataset,removedVars)
        
    evalDir = opts.o

    from DeepJetCore.DataCollection import DataCollection
    testd=DataCollection()
    testd.readFromFile(inputTestDataCollection)

    if os.path.isdir(evalDir):
        raise Exception('output directory: %s must not exists yet' %evalDir)
    else:
        os.mkdir(evalDir)

    df, features_val = makePlots(testd, evalModel, evalDir)
    makeLossPlot(trainDir,evalDir)

    #df = evaluate(testd, inputTrainDataCollection, evalModel, evalDir)
    #make_plots(evalDir, savedir='Plots')
                                                       
