import sys, os
from argparse import ArgumentParser
import setGPU
os.environ['CUDA_VISIBLE_DEVICES']=''
# Options 
parser = ArgumentParser(description ='Script to run the training and evaluate it')
parser.add_argument("--adv", action='store_true', default=False, help="Load adversarial model")
parser.add_argument("--decor", action='store_true', default=False, help="Serve decorrelated training targets")
parser.add_argument("--multi", action='store_true', default=False, help="Binary or categorical crossentropy")
parser.add_argument("-i", help="Training dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-t", help="Testing dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-d",  help="Training output dir", default=None, metavar="PATH")
parser.add_argument("-o",  help="Eval output dir", default=None, metavar="PATH")
parser.add_argument("-p",  help="Plot output dir within Eval output dir", default="Plots", metavar="PATH")
parser.add_argument("--storeInputs", action='store_true', help="Store inputs in df", default=False)
parser.add_argument("--taggerName",  help="DeepDouble{input} name in ROC plots", default="X")
opts=parser.parse_args()
if opts.decor:  os.environ['DECORRELATE'] = "True"
else:  os.environ['DECORRELATE'] = "False"

sampleDatasets_pf_cpf_sv = ["db","pf","cpf","sv"]
sampleDatasets_cpf_sv = ["db","cpf","sv"]
sampleDatasets_sv = ["db","sv"]

#select model and eval functions
if opts.adv:
    from models import model_DeepDoubleXAdversarial as trainingModel
else:
    from models import model_DeepDoubleXReference as trainingModel
from DeepJetCore.training.training_base import training_base
from eval_functions import loadModel, evaluate
from plots_from_df import make_plots
from Metrics import global_metrics_list, acc_kldiv


inputDataset = sampleDatasets_cpf_sv
trainDir = opts.d
inputTrainDataCollection = opts.t
inputTestDataCollection = opts.i
LoadModel = False
removedVars = None

if True:
    evalModel = loadModel(trainDir,inputTrainDataCollection,trainingModel,LoadModel,inputDataset,removedVars,adv=opts.adv)
    evalDir = opts.o

    from DeepJetCore.DataCollection import DataCollection
    testd=DataCollection()
    testd.readFromFile(inputTestDataCollection)

    if os.path.isdir(evalDir):
        raise Exception('output directory: %s must not exists yet' %evalDir)
    else:
        os.mkdir(evalDir)

    df = evaluate(testd, inputTrainDataCollection, evalModel, evalDir, storeInputs=opts.storeInputs, adv=opts.adv)
    make_plots(evalDir, savedir=opts.p, taggerName=opts.taggerName)


