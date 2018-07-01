import sys, os
from argparse import ArgumentParser
                                                                                                                                  
# Options 
parser = ArgumentParser(description ='Script to run the training and evaluate it')
parser.add_argument("--multi", action='store_true', default=False, help="Binary or categorical crossentropy")
#parser.add_argument("-g", "--gpu", default=1, help="Run on gpu's (Need to be in deepjetLinux3_gpu env)", const=1)
parser.add_argument("-i", help="Training dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-o",  help="Training output dir", default=None, metavar="PATH")
parser.add_argument("--batch",  help="Batch size, default = 2000", default=2000, metavar="INT")
parser.add_argument("--epochs",  help="Epochs, default = 50", default=50, metavar="INT")
parser.add_argument("--resume", action='store_true', default=False, help="Continue previous")
opts=parser.parse_args()

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

#select model and eval functions
from models.DeepJet_models_final import conv_model_final as trainingModel
from DeepJetCore.training.training_base import training_base
from eval_functions import loadModel

inputDataset = sampleDatasets_pf_cpf_sv

if True:
    args = MyClass()
    args.inputDataCollection = opts.i
    args.outputDir = opts.o
    
    multi_gpu = len([x for x in os.popen("nvidia-smi -L").read().split("\n") if "GPU" in x])
    print "nGPU:", multi_gpu
	

    train=training_base(splittrainandtest=0.9,testrun=False, resumeSilently=opts.resume, parser=args)
    #train=training_base(splittrainandtest=0.9,testrun=False, args=args)
    if not train.modelSet():
        train.setModel(trainingModel, 
			#num_classes=5, #num_regclasses=5,
			datasets=inputDataset, 
			multi_gpu=multi_gpu
			)
    	if opts.multi: 
		loss = 'categorical_crossentropy'
		accuracy = 'categorical_accuracy'
	else: 
		loss = 'binary_crossentropy'
		accuracy = 'binary_accuracy'

        train.compileModel(learningrate=0.001,
                        	loss=[loss],
	                        metrics=['accuracy','binary_accuracy','MSE','MSLE'],
				loss_weights=[1.])

    model, history = train.trainModel(nepochs=int(opts.epochs),
						batchsize=int(opts.batch),
                                                stop_patience=1000,
                                                lr_factor=0.7,
                                                lr_patience=10,
                                                lr_epsilon=0.00000001,
                                                lr_cooldown=2,
                                                lr_minimum=0.00000001,
                                                maxqsize=100)

    
