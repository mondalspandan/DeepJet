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
    y = y_in[:,NBINS:NBINS+2]

    return categorical_accuracy(y, x)

def mass_kldiv_q(y_in,x):
    """
    KL divergence term for anti-tag events (QCD) to be used with custom loss_kldiv 
    """
    h = y_in[:,0:NBINS]
    y = y_in[:,NBINS:NBINS+2]
    h_all = K.dot(K.transpose(h), y)
    h_all_q = h_all[:,0]
    h_all_h = h_all[:,1]
    h_all_q = h_all_q / K.sum(h_all_q,axis=0)
    h_all_h = h_all_h / K.sum(h_all_h,axis=0)
    h_btag_anti_q = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
    h_btag_anti_h = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
    h_btag_q = h_btag_anti_q[:,1]
    h_btag_q = h_btag_q / K.sum(h_btag_q,axis=0)
    h_anti_q = h_btag_anti_q[:,0]
    h_anti_q = h_anti_q / K.sum(h_anti_q,axis=0)
    h_btag_h = h_btag_anti_h[:,1]
    h_btag_h = h_btag_h / K.sum(h_btag_h,axis=0)
    h_anti_h = h_btag_anti_h[:,0]
    h_anti_h = h_anti_h / K.sum(h_anti_h,axis=0)

    return kullback_leibler_divergence(h_btag_q, h_anti_q)

def mass_kldiv_h(y_in,x):
    """
    KL divergence term for tag events (H) to be used with custom loss_kldiv 
    """
    h = y_in[:,0:NBINS]
    y = y_in[:,NBINS:NBINS+2]
    h_all = K.dot(K.transpose(h), y)
    h_all_q = h_all[:,0]
    h_all_h = h_all[:,1]
    h_all_q = h_all_q / K.sum(h_all_q,axis=0)
    h_all_h = h_all_h / K.sum(h_all_h,axis=0)
    h_btag_anti_q = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
    h_btag_anti_h = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
    h_btag_q = h_btag_anti_q[:,1]
    h_btag_q = h_btag_q / K.sum(h_btag_q,axis=0)
    h_anti_q = h_btag_anti_q[:,0]
    h_anti_q = h_anti_q / K.sum(h_anti_q,axis=0)
    h_btag_h = h_btag_anti_h[:,1]
    h_btag_h = h_btag_h / K.sum(h_btag_h,axis=0)
    h_anti_h = h_btag_anti_h[:,0]
    h_anti_h = h_anti_h / K.sum(h_anti_h,axis=0)

    return kullback_leibler_divergence(h_btag_h, h_anti_h)


#please always register the loss function here
global_metrics_list['acc_kldiv']=acc_kldiv
global_metrics_list['mass_kldiv_q']=mass_kldiv_q
global_metrics_list['mass_kldiv_h']=mass_kldiv_h



