from keras.layers import Dense, Dropout, Flatten,Concatenate, Convolution2D, LSTM,merge, Convolution1D, Conv2D, GRU, SpatialDropout1D, Conv1D, Lambda
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import Add, Multiply
from buildingBlocks import block_deepFlavourConvolutions, block_deepFlavourDense, block_SchwartzImage, block_deepFlavourBTVConvolutions

def model_deepFlavourNoNeutralReference(Inputs,nclasses,nregclasses,dropoutRate=0.1):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = BatchNormalization(momentum=0.6,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=0.6,name='cpf_input_batchnorm')     (Inputs[1])
    vtx    =     BatchNormalization(momentum=0.6,name='vtx_input_batchnorm')     (Inputs[2])
    ptreginput = BatchNormalization(momentum=0.6,name='reg_input_batchnorm')     (Inputs[3])
    
    cpf, vtx = block_deepFlavourBTVConvolutions(
        charged=cpf,
        vertices=vtx,
        dropoutRate=dropoutRate,
        active=True,
        batchnorm=True
        )
    
    
    #
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf=BatchNormalization(momentum=0.6,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=0.6,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,vtx ])
    
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=0.6)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[1])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[2])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[3])
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    #
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def convolutional_model_deepcsv(Inputs,nclasses,nregclasses,dropoutRate=-1):
    
    cpf=Inputs[1]
    vtx=Inputs[2]
    
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    
    
    vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    
    cpf=Flatten()(cpf)
    vtx=Flatten()(vtx)
        
    x = Concatenate()( [Inputs[0],cpf,vtx ])
        
    x  = block_deepFlavourDense(x,dropoutRate)

    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_broad(Inputs,nclasses,nregclasses,dropoutRate=-1):
    """
    reference 1x1 convolutional model for 'deepFlavour', as for DPS note
    """  
   
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    image = block_SchwartzImage(image=Inputs[4],dropoutRate=dropoutRate,active=False)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx,image ])
    
    x  = block_deepFlavourDense(x,dropoutRate)

    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_broad_map(Inputs,nclasses,nregclasses,dropoutRate=-1):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    """  
    
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    image = block_SchwartzImage(image=Inputs[4],dropoutRate=dropoutRate)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx,image ])
    
    x  = block_deepFlavourDense(x,dropoutRate)
    
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def convolutional_model_broad_map_reg(Inputs,nclasses,nregclasses,dropoutRate):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    """  
    
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    image = block_SchwartzImage(image=Inputs[4],dropoutRate=dropoutRate)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx,image,Inputs[5] ])
    
    x  = block_deepFlavourDense(x,dropoutRate)
    
    predictions = [Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x),
                   Dense(nregclasses, activation='linear',kernel_initializer='ones',name='E_pred')(x)]
    model = Model(inputs=Inputs, outputs=predictions)
    return model




def convolutional_model_broad_reg(Inputs,nclasses,nregclasses,dropoutRate=-1):
    """
    the inputs are really not working as they are. need a reshaping well before
    """  
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx,Inputs[4] ])
    
    x  = block_deepFlavourDense(x,dropoutRate)
    
    
    predictions = [Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x),
                   Dense(nregclasses, activation='linear',kernel_initializer='ones',name='E_pred')(x)]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_broad_reg2(Inputs,nclasses,nregclasses,dropoutRate=-1):
    """
    Flavour tagging and regression in one model. Fully working
    """
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Flatten()(cpf)
    
    
    npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[2])
    npf = Dropout(dropoutRate)(npf)
    npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    npf = Dropout(dropoutRate)(npf)
    npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    npf = Flatten()(npf)
    
    vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[3])
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Flatten()(vtx)
    
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx ] )
    

    x=  Dense(350, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    flav=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    
    
    
    
    ptcpf  = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    ptcpf = Dropout(dropoutRate)(ptcpf)
    ptcpf = Flatten()(ptcpf)
    ptnpf  = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[2])
    ptnpf = Dropout(dropoutRate)(ptnpf)
    ptnpf = Flatten()(ptnpf)
   
    xx=Concatenate()( [Inputs[4],flav,ptcpf,ptnpf] )
    xx=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(xx)
    
    ptandsigma=Dense(2, activation='linear',kernel_initializer='lecun_uniform')(xx)
    
    predictions = [flav,ptandsigma]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_lessbroad(Inputs,nclasses,nregclasses,dropoutRate=-1):
    """
    the inputs are really not working as they are. need a reshaping well before
    """
   
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform',input_shape=Inputshapes[0])(Inputs[0])
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform')(gl)
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform')(gl)
    
    
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf = Flatten()(cpf)
    
    
    npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[2])
    npf = Dropout(dropoutRate)(npf)
    npf = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    npf = Dropout(dropoutRate)(npf)
    npf = Flatten()(npf)
    
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[3])
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Flatten()(vtx)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx ] )
    x = Dropout(dropoutRate)(x)

    x=  Dense(600, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def convolutional_model_ConvCSV(Inputs,nclasses,nregclasses,dropoutRate=0.25):
    """
    Inputs similar to 2016 training, but with covolutional layers on each track and sv
    """
    
    a  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    a = Dropout(dropoutRate)(a)
    a  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(a)
    a = Dropout(dropoutRate)(a)
    a  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(a)
    a = Dropout(dropoutRate)(a)
    a=Flatten()(a)
    
    c  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[2])
    c = Dropout(dropoutRate)(c)
    c  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(c)
    c = Dropout(dropoutRate)(c)
    c  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(c)
    c = Dropout(dropoutRate)(c)
    c=Flatten()(c)
    
    x = Concatenate()( [Inputs[0],a,c] )
    
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def FC(data, num_hidden, act='relu', p=None, name='', trainable=True):
    kernel_initializer_fc = 'lecun_uniform'
    if act=='leakyrelu':
        fc = Dense(num_hidden, activation='linear',
                   name='%s_%s' % (name,act),
                   kernel_initializer=kernel_initializer_fc,
                   trainable=trainable)(data) # Add any layer, with the default of a linear squashing function
        fc = LeakyReLU(alpha=.001)(fc)   # add an advanced activation                                                                                                     
    else:
        fc = Dense(num_hidden, activation=act,
                   name='%s_%s' % (name,act),
                   kernel_initializer=kernel_initializer_fc,
                   trainable=trainable)(data)
    if not p:
        return fc
    else:
        dropout = Dropout(rate=p, name='%s_dropout' % name)(fc)
        return dropout

def model_DeepDoubleXReference(inputs, num_classes, num_regclasses, datasets = ['db','cpf','sv'], removedVars = None, multi_gpu=1,  **kwargs):
    """
    reference 1x1 convolutional model for 'DeepDoubleX'
    with recurrent layers and batch normalisation
    """
    kernel_initializer = 'he_normal'
    kernel_initializer_fc = 'lecun_uniform'
    normalizedInputs = []

    for i in range(len(inputs)):
        normedLayer = BatchNormalization(momentum=0.3,name = '%s_input_batchnorm'%datasets[i])(inputs[i])
        normalizedInputs.append(normedLayer)

    flattenLayers = []
    flattenLayers.append(Flatten()(normalizedInputs[0]))

    for ds, normalizedInput in zip(datasets[1:],normalizedInputs[1:]):                
        x = Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same', 
                                kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv1'%ds, 
                                activation = 'relu')(normalizedInput)
        x = SpatialDropout1D(rate=0.1)(x)
        x = Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same', 
                             kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv2'%ds, 
                             activation = 'relu')(x)
        x = SpatialDropout1D(rate=0.1)(x)
        x = GRU(50,go_backwards=True,implementation=2,name='%s_gru'%ds)(x)
        x = Dropout(rate=0.1)(x)
        flattenLayers.append(x)

    concat = Concatenate()(flattenLayers)


    fc = FC(concat, 100, p=0.1, name='fc1')
    output = Dense(num_classes, activation='softmax', name='ID_pred', kernel_initializer=kernel_initializer_fc)(fc)
                            
    model = Model(inputs=inputs, outputs=[output])
    if multi_gpu > 1:
        from multi_gpu_model import multi_gpu_model
        model = multi_gpu_model(model, gpus=multi_gpu)

    return model

def model_DeepDoubleBLambda(inputs, num_classes, num_regclasses, datasets = ['db','pf','cpf','sv'], removedVars = None):
    """
    lambda 1x1 convolutional model for 'DeepDoubleB'
    with recurrent layers and batch normalisation
    """
    kernel_initializer = 'he_normal'
    kernel_initializer_fc = 'lecun_uniform'
    normalizedInputs = []

    for i in range(len(inputs)):
        normedLayer = BatchNormalization(momentum=0.3,name = '%s_input_batchnorm'%datasets[i])(inputs[i])
        normalizedInputs.append(normedLayer)

    lambdaLayer = Lambda(lambda x: 0*x, name='%s_lambda'%datasets[0])(normalizedInputs[0])
    def slicer(x):
        return x[:,:,0:1]    
    lambdaLayer = Lambda(slicer)(lambdaLayer)
    flattenLayers = []
    flattenLayers.append(Flatten()(lambdaLayer))

    for ds, normalizedInput in zip(datasets[1:],normalizedInputs[1:]):                
        x = Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same', 
                                kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv1'%ds, 
                                activation = 'relu')(normalizedInput)
        x = SpatialDropout1D(rate=0.1)(x)
        x = Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same', 
                             kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv2'%ds, 
                             activation = 'relu')(x)
        x = SpatialDropout1D(rate=0.1)(x)
        x = GRU(50,go_backwards=True,implementation=2,name='%s_gru'%ds)(x)
        x = Dropout(rate=0.1)(x)
        flattenLayers.append(x)

    concat = Concatenate()(flattenLayers)


    fc = FC(concat, 100, p=0.1, name='fc1')
    output = Dense(num_classes, activation='softmax', name='ID_pred', kernel_initializer=kernel_initializer_fc)(fc)
                            
    model = Model(inputs=inputs, outputs=[output])

    return model

