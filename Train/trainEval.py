import sys, os
from argparse import ArgumentParser
                                                                                                                                  
import tensorflow as tf
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').disabled = True
logging.getLogger("tensorflow").setLevel(logging.WARNING)

# Options 
parser = ArgumentParser(description ='Script to run the training and evaluate it')
parser.add_argument("--train", action='store_true', default=False, help="Run training")
parser.add_argument("--eval", action='store_true', default=False, help="Run evaluation")
parser.add_argument("--multi", action='store_true', default=False, help="Binary or categorical crossentropy")
#parser.add_argument("-g", "--gpu", default=1, help="Run on gpu's (Need to be in deepjetLinux3_gpu env)", const=1)
parser.add_argument("-i", help="Training dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-n",  help="Training output dir", default=None, metavar="PATH")
parser.add_argument("-o",  help="Training output dir", default=None, metavar="PATH")
opts=parser.parse_args()
# Toggle training or eval
TrainBool = opts.train
EvalBool = opts.eval
multi = opts.multi

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

#removedVars = [[],range(0,10), range(0,22),[0,1,2,3,4,5,6,7,8,9,10,13]]
removedVars = None

#Toggle to load model directly (True) or load weights (False) 
LoadModel = True

#select model and eval functions
from models.DeepJet_models_final import conv_model_final as trainingModel
from DeepJetCore.training.training_base import training_base
from eval_funcs import loadModel, makeRoc, _byteify, makeLossPlot, makeComparisonPlots


trainDir = opts.n
inputTrainDataCollection = opts.i
inputTestDataCollection = opts.i
inputDataset = sampleDatasets_pf_cpf_sv

if TrainBool:
    args = MyClass()
    args.inputDataCollection = inputTrainDataCollection
    args.outputDir = trainDir
    args.multi_gpu = os.system("nvidia-smi -L")+1
    print "nGPU:", args.multi_gpu

    train=training_base(splittrainandtest=0.9,testrun=False, parser=args)
    #train=training_base(splittrainandtest=0.9,testrun=False, args=args)
    if not train.modelSet():
        train.setModel(trainingModel, 
			 #num_classes=5, #num_regclasses=5,
				datasets=inputDataset, )
    	if multi:
		loss = 'categorical_crossentropy'
		metric = 'categorical_accuracy'
	else:
		metric = 'binary_accuracy'
		loss = 'binary_crossentropy'
        train.compileModel(learningrate=0.001,
                           loss=[loss],
                           	 metrics=[metric ,'MSE','MSLE'],
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
        model, history = train.trainModel(nepochs=50,
                                                   batchsize=2000,
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

    df = makeRoc(testd, evalModel, evalDir, False)
    makeLossPlot(trainDir,evalDir)
    
