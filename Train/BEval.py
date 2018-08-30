import sys, os
from argparse import ArgumentParser
                                                                                                                                  
# Options 
parser = ArgumentParser(description ='Script to run the training and evaluate it')
parser.add_argument("--multi", action='store_true', default=False, help="Binary or categorical crossentropy")
parser.add_argument("-i", help="Training dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-t", help="Testing dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-d",  help="Training output dir", default=None, metavar="PATH")
parser.add_argument("-o",  help="Eval output dir", default=None, metavar="PATH")
parser.add_argument("-p",  help="Plot output dir within Eval output dir", default="Plots", metavar="PATH")
parser.add_argument("--storeInputs", action='store_true', help="Store inputs in df", default=False)
opts=parser.parse_args()

sampleDatasets_pf_cpf_sv = ["db","pf","cpf","sv"]
sampleDatasets_cpf_sv = ["db","cpf","sv"]
sampleDatasets_sv = ["db","sv"]

#select model and eval functions
from models.DeepJet_models_final import conv_model_final as trainingModel
from DeepJetCore.training.training_base import training_base
from eval_functions import loadModel, evaluate
from plots_from_df import make_plots

inputDataset = sampleDatasets_pf_cpf_sv
trainDir = opts.d
inputTrainDataCollection = opts.t
inputTestDataCollection = opts.i
LoadModel = True
removedVars = None

if True:
    evalModel = loadModel(trainDir,inputTrainDataCollection,trainingModel,LoadModel,inputDataset,removedVars)
    evalDir = opts.o

    from DeepJetCore.DataCollection import DataCollection
    testd=DataCollection()
    testd.readFromFile(inputTestDataCollection)

    if os.path.isdir(evalDir):
        raise Exception('output directory: %s must not exists yet' %evalDir)
    else:
        os.mkdir(evalDir)

    df = evaluate(testd, inputTrainDataCollection, evalModel, evalDir, storeInputs=opts.storeInputs)
    make_plots(evalDir, savedir=opts.p)


