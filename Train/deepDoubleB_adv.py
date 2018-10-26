import os, sys
from argparse import ArgumentParser
parser = ArgumentParser('Run the training')
parser.add_argument('inputDataCollection')
parser.add_argument('outputDir')
parser.add_argument('--modelMethod', help='Method to be used to instantiate model in derived training class', metavar='OPT', default=None)
parser.add_argument("--gpu",  help="select specific GPU",   type=int, metavar="OPT", default=-1)
parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=-1)
parser.add_argument("--decor", action='store_true', default=True, help="Use kl_div to decorrelate")
parser.add_argument("--lambda-adv", default='15', help="lambda for adversarial training", type=str)
args=parser.parse_args()
if args.decor:
    os.environ['DECORRELATE'] = "True"
else:
    os.environ['DECORRELATE'] = "False"
os.environ['LAMBDA_ADV'] = args.lambda_adv
from DeepJetCore.training.training_base import training_base
from Losses import loss_NLL, loss_meansquared, loss_kldiv, loss_reg, global_loss_list, NBINS, loss_disc, loss_adv, LAMBDA_ADV
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights
from Layers import global_layers_list
from Metrics import global_metrics_list, acc_reg, mass_kldiv_q, mass_kldiv_h
custom_objects_list = {}
custom_objects_list.update(global_loss_list)
custom_objects_list.update(global_layers_list)
custom_objects_list.update(global_metrics_list)

print("using LAMBDA_ADV =",LAMBDA_ADV)
#also does all the parsing
train=training_base(testrun=False,renewtokens=False,parser=args)

trainedModel = 'train_deepDoubleB_reference/KERAS_check_best_model.h5'

if not train.modelSet():
    from models import model_DeepDoubleXAdversarial as trainingModel


    train.setModel(trainingModel, datasets=['db','cpf','sv'], removedVars=None,
                   nRegTargets=NBINS,
                   discTrainable=True,
                   advTrainable=True)

    train.compileModel(learningrate=0.00001,
                       loss=[loss_reg],
                       metrics=[acc_reg, mass_kldiv_q, mass_kldiv_h, loss_disc, loss_adv],
                       loss_weights=[1.])

    train.loadWeights(trainedModel)

    train.keras_model.summary()
    train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')
    model,history,callbacks = train.trainModel(nepochs=200,
                                               batchsize=2048,
                                               stop_patience=50,
                                               lr_factor=0.5,
                                               lr_patience=10,
                                               lr_epsilon=0.00000001,
                                               lr_cooldown=2,
                                               lr_minimum=0.00000001,
                                               maxqsize=100)
    
