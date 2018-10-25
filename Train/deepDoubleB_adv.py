import os
os.environ['DECORRELATE'] = "True"
from DeepJetCore.training.training_base import training_base
from Losses import loss_NLL, loss_meansquared, loss_kldiv, loss_reg, global_loss_list, NBINS, LAMBDA_ADV, loss_disc, loss_adv
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights
from Layers import global_layers_list
from Metrics import global_metrics_list, acc_reg, mass_kldiv_q, mass_kldiv_h
custom_objects_list = {}
custom_objects_list.update(global_loss_list)
custom_objects_list.update(global_layers_list)
custom_objects_list.update(global_metrics_list)

#also does all the parsing
train=training_base(testrun=False,renewtokens=False)

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
    
