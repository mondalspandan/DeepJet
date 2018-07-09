

from DeepJetCore.training.training_base import training_base
from Losses import loss_NLL, loss_meansquared   
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights

#also does all the parsing
train=training_base(testrun=True,renewtokens=False)

if not train.modelSet():
    from models import model_deepDoubleBReference as trainingModel

    train.setModel(trainingModel, datasets=['db','cpf','SV'], removedVars=None)
    
    train.compileModel(learningrate=0.001,
                       loss=['categorical_crossentropy'],
                       metrics=['accuracy'],
                       loss_weights=[1.])
    
    model,history,callbacks = train.trainModel(nepochs=200,
                                               batchsize=128,
                                               stop_patience=50,
                                               lr_factor=0.5,
                                               lr_patience=10,
                                               lr_epsilon=0.00000001,
                                               lr_cooldown=2,
                                               lr_minimum=0.00000001,
                                               maxqsize=5)
    
