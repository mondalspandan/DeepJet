from keras import backend as K
#from tensorflow import where, greater, abs, zeros_like, exp
import tensorflow as tf
from keras.metrics import categorical_accuracy
from keras.losses import kullback_leibler_divergence

global_metrics_list = {}

from Losses import NBINS, MMAX, MMIN

def acc_kldiv(y_in,x):
    """
    Corrected accuracy to be used with custom loss_kldiv
    """
    h = y_in[:,0:NBINS]
    y = y_in[:,NBINS:]

    return categorical_accuracy(y, x)

def acc_reg(y_in,x_in):
    """
    Corrected accuracy to be used with custom loss_reg
    """
    h = y_in[:,0:NBINS]
    y = y_in[:,NBINS:]
    hpred = x_in[:,0:NBINS]
    ypred = x_in[:,NBINS:]

    return categorical_accuracy(y, ypred)

def mass_kldiv_q(y_in,x):
    """
    KL divergence term for anti-tag events (QCD) to be used with custom loss_kldiv 
    """
    h = y_in[:,0:NBINS]
    y = y_in[:,NBINS:NBINS+2]
    # build mass histogram for true q events weighted by q, b prob
    h_alltag_q = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
    
    # select mass histogram for true q events weighted by q prob; normalize
    h_qtag_q = h_alltag_q[:,0]
    h_qtag_q = h_qtag_q / K.sum(h_qtag_q,axis=0)
    # select mass histogram for true q events weighted by b prob; normalize
    h_btag_q = h_alltag_q[:,1]
    h_btag_q = h_btag_q / K.sum(h_btag_q,axis=0)

    return kullback_leibler_divergence(h_btag_q, h_qtag_q)

def mass_jsdiv_q(y_in,x):
    """
    KL divergence term for anti-tag events (QCD) to be used with custom loss_kldiv 
    """
    h = y_in[:,0:NBINS]
    y = y_in[:,NBINS:NBINS+2]
    # build mass histogram for true q events weighted by q, b prob
    h_alltag_q = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
    
    # select mass histogram for true q events weighted by q prob; normalize
    h_qtag_q = h_alltag_q[:,0]
    h_qtag_q = h_qtag_q / K.sum(h_qtag_q,axis=0)
    # select mass histogram for true q events weighted by b prob; normalize
    h_btag_q = h_alltag_q[:,1]
    h_btag_q = h_btag_q / K.sum(h_btag_q,axis=0)

    h_aver_q = 0.5*h_btag_q+0.5*h_qtag_q
    return 0.5*kullback_leibler_divergence(h_btag_q, h_aver_q) + 0.5*kullback_leibler_divergence(h_qtag_q, h_aver_q) 

def mass_kldiv_h(y_in,x):
    """
    KL divergence term for tag events (H) to be used with custom loss_kldiv 
    """
    h = y_in[:,0:NBINS]
    y = y_in[:,NBINS:NBINS+2]

    # build mass histogram for true b events weighted by q, b prob
    h_alltag_b = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
    
    # select mass histogram for true b events weighted by q prob; normalize        
    h_qtag_b = h_alltag_b[:,0]
    h_qtag_b = h_qtag_b / K.sum(h_qtag_b,axis=0)
    # select mass histogram for true b events weighted by b prob; normalize        
    h_btag_b = h_alltag_b[:,1]
    h_btag_b = h_btag_b / K.sum(h_btag_b,axis=0)
    
    return kullback_leibler_divergence(h_btag_b, h_qtag_b)


#please always register the loss function here
global_metrics_list['acc_kldiv']=acc_kldiv
global_metrics_list['mass_kldiv_q']=mass_kldiv_q
global_metrics_list['mass_kldiv_h']=mass_kldiv_h
global_metrics_list['acc_reg']=acc_reg
global_metrics_list['mass_jsdiv_q']=mass_jsdiv_q



