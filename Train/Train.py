import sys, os
from argparse import ArgumentParser
                                                                                                                                  
# Options 
parser = ArgumentParser(description ='Script to run the training and evaluate it')
parser.add_argument("--decor", action='store_true', default=False, help="Use kl_div to decorrelate")
parser.add_argument("--adv", action='store_true', default=False, help="Use adversarial network to decorrelate")
parser.add_argument("--lambda-adv", default='15', help="lambda for adversarial training", type=str)
parser.add_argument("--classes", help="Number of classes, default 2", default=2, metavar="INT")
parser.add_argument("-g", "--gpu", default=-1, help="Use 1 specific gpu (Need to be in deepjetLinux3_gpu env)", type=int)
parser.add_argument("-m", "--multi-gpu", action='store_true', default=False, help="Use all visible gpus (Need to be in deepjetLinux3_gpu env)")
parser.add_argument("-i", help="Training dataCollection.dc", default=None, metavar="FILE")
parser.add_argument("-o",  help="Training output dir", default=None, metavar="PATH")
parser.add_argument("--batch",  help="Batch size, default = 2000", default=2000, metavar="INT")
parser.add_argument("--epochs",  help="Epochs, default = 50", default=50, metavar="INT")
parser.add_argument("--resume", action='store_true', default=False, help="Continue previous")
parser.add_argument("--pretrained-model", default=None, help="start from specified pretrained model", type=str)
opts=parser.parse_args()
if opts.decor:  os.environ['DECORRELATE'] = "True"
else:  os.environ['DECORRELATE'] = "False"
os.environ['LAMBDA_ADV'] = opts.lambda_adv

# Some used default settings
class MyClass:
    """A simple example class"""
    def __init__(self):
	self.gpu = -1
	self.gpufraction = -1
	self.modelMethod = ''
        self.inputDataCollection = ''
        self.outputDir = ''

sampleDatasets_cpf_sv = ["db","cpf","sv"]

#select model and eval functions
if opts.adv:
    from models.convolutional import model_DeepDoubleXAdversarial as trainingModel
else:
    from models.convolutional import model_DeepDoubleXReference  as trainingModel
from DeepJetCore.training.training_base import training_base

from Losses import loss_NLL, loss_meansquared, loss_kldiv, loss_kldiv_3class, global_loss_list, custom_crossentropy, loss_reg, NBINS, loss_disc, loss_adv, LAMBDA_ADV
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights
from Layers import global_layers_list
from Metrics import global_metrics_list, acc_kldiv, mass_kldiv_q, mass_kldiv_h, acc_reg
custom_objects_list = {}
custom_objects_list.update(global_loss_list)
custom_objects_list.update(global_layers_list)
custom_objects_list.update(global_metrics_list)

inputDataset = sampleDatasets_cpf_sv

if True:  # Should probably fix
    args = MyClass()
    args.inputDataCollection = opts.i
    args.outputDir = opts.o

    multi_gpu = 1
    if opts.multi_gpu:
        # use all visible GPUs
        multi_gpu = len([x for x in os.popen("nvidia-smi -L").read().split("\n") if "GPU" in x])
        args.gpu = ','.join([str(i) for i in range(multi_gpu)])
        print(args.gpu)

    # Separate losses and metrics for training and decorrelation
    if opts.decor and opts.adv:
        print("INFO: using LAMBDA_ADV = %f"%LAMBDA_ADV)
        loss = loss_reg
        metrics = [acc_reg, mass_kldiv_q, mass_kldiv_h, loss_disc, loss_adv]
    elif opts.decor: 
        nClasses=int(opts.classes)
        if nClasses==2:
            loss = loss_kldiv
        elif nClasses==3:
            loss = loss_kldiv_3class
        else:
            print("WARNING: Decorrelation with %d classes is not supported yet, using loss for 2 classes." %nClasses)
            loss = loss_kldiv
        print("# of classes for decorrelation: %d" %nClasses)
        metrics=[acc_kldiv, mass_kldiv_q, mass_kldiv_h]
    else: 
        loss = 'categorical_crossentropy'
        metrics=['accuracy']

    # Set up training
    train=training_base(splittrainandtest=0.9,testrun=False, useweights=True, resumeSilently=opts.resume, renewtokens=False, parser=args)
    if not train.modelSet():
        train.setModel(trainingModel, 
			datasets=inputDataset, 
			multi_gpu=multi_gpu
			)
        train.compileModel(learningrate=0.001,
                        	loss=[loss],
	                        metrics=metrics,
				loss_weights=[1.])

        try: # To make sure no training is lost on restart
	    train.loadWeights(opts.o+"/KERAS_check_best_model.h5")
        except:
	    pass

    # Pretrain and fix batch normalization weights to be consistent
    if not opts.decor:
	model, history = train.trainModel(nepochs=1, 
                                          batchsize=int(opts.batch), 
                                          stop_patience=50,
                                          lr_factor=0.7, 
                                          lr_patience=10, 
                                          lr_epsilon=0.00000001, 
                                          lr_cooldown=2, 
                                          lr_minimum=0.00000001, 
                                          maxqsize=100)

        train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')
	
        # Need to recompile after fixing batchnorm weights or loading model and changing loss to decorrelate
        train.compileModel(learningrate=0.001,
                           loss=[loss],
	                   metrics=metrics,
		           loss_weights=[1.])
        # Load best weights
        try: # To make sure no training is lost on restart
	    train.loadWeights(opts.o+"/KERAS_check_best_model.h5")
        except:
	    pass

    # Override by loading pretrained model
    if opts.pretrained_model is not None:
        train.loadWeights(opts.pretrained_model)
        
    # Train
    model, history = train.trainModel(nepochs=int(opts.epochs),
				      batchsize=int(opts.batch),
                                      stop_patience=50,
                                      lr_factor=0.7,
                                      lr_patience=10,
                                      lr_epsilon=0.00000001,
                                      lr_cooldown=2,
                                      lr_minimum=0.00000001,
                                      maxqsize=100)

    
