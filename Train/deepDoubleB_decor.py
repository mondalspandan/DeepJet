

from DeepJetCore.training.training_base import training_base
from Losses import loss_NLL, loss_meansquared, loss_kldiv, global_loss_list
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights
from Layers import global_layers_list
from Metrics import global_metrics_list
custom_objects_list = {}
custom_objects_list.update(global_loss_list)
custom_objects_list.update(global_layers_list)
custom_objects_list.update(global_metrics_list)

#also does all the parsing
train=training_base(testrun=True,renewtokens=False)

trainedModel = '/data/shared/BumbleB/DDBfull100/training/KERAS_check_best_model.h5'

if not train.modelSet():
    from models import model_deepDoubleBReference as trainingModel

    train.setModel(trainingModel, datasets=['db','pf','cpf'], removedVars=None)
    
    train.compileModel(learningrate=0.001,
                       loss=[loss_kldiv],
                       metrics=['accuracy'],
                       loss_weights=[1.])

    train.loadWeights(trainedModel)
    train.keras_model.summary()

    model,history,callbacks = train.trainModel(nepochs=200,
                                               batchsize=2048,
                                               stop_patience=50,
                                               lr_factor=0.5,
                                               lr_patience=10,
                                               lr_epsilon=0.00000001,
                                               lr_cooldown=2,
                                               lr_minimum=0.00000001,
                                               maxqsize=100)
    
