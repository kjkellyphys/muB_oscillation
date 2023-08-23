import pickle
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate

from MicroTools import *

#################################### BEGIN MINIBOONE ANALYSIS ######################################

def GetRescaledCovarianceMatrix(fractional_covariance, predicted_signal, predicted_background, predicted_numu):
    signal_prediction_matrix = np.diag(np.concatenate([predicted_signal,predicted_background*0.0,predicted_numu*0.0]))
    total_prediction_matrix = np.diag(np.concatenate([predicted_signal,predicted_background,predicted_numu]))
    # miniboone covariance matrix is fractional, this rescale to physical rates
    rescaled_cov_matrix = np.dot(total_prediction_matrix,np.dot(fractional_covariance,total_prediction_matrix))
    return (rescaled_cov_matrix + signal_prediction_matrix) # this adds the statistical error on data


def CalculateChi2(fractional_covariance,
                  predicted_signal, 
                  predicted_background, 
                  predicted_numu,
                  observed_nue, 
                  observed_numu):

    # by_hand_mod = np.append(predicted_signal[:4],predicted_signal[4:]*50)
    by_hand_mod = predicted_signal

    # get rescaled covariance
    rescaled_covariance = GetRescaledCovarianceMatrix(fractional_covariance,
                                                      by_hand_mod, predicted_background, predicted_numu)
    # collapse background part of the covariance
    n_signal = len(predicted_signal)
    n_background = len(predicted_background)
    assert(n_signal == n_background)
    n_numu = len(predicted_numu)

    covariance = np.zeros([n_signal+n_numu,n_signal+n_numu])

    covariance[0:n_signal,0:n_signal] = rescaled_covariance[0:n_signal,0:n_signal] + rescaled_covariance[n_signal:2*n_signal,0:n_signal] + rescaled_covariance[0:n_signal,n_signal:2*n_signal] + rescaled_covariance[n_signal:2*n_signal,n_signal:2*n_signal]
    covariance[n_signal:(n_signal+n_numu),0:n_signal] = rescaled_covariance[2*n_signal:(2*n_signal+n_numu),0:n_signal] + rescaled_covariance[2*n_signal:(2*n_signal+n_numu),n_signal:2*n_signal]
    covariance[0:n_signal,n_signal:(n_signal+n_numu)] = rescaled_covariance[0:n_signal,2*n_signal:(2*n_signal+n_numu)] + rescaled_covariance[n_signal:2*n_signal,2*n_signal:(2*n_signal+n_numu)]
    covariance[n_signal:(n_signal+n_numu),n_signal:(n_signal+n_numu)] = rescaled_covariance[2*n_signal:2*n_signal+n_numu,2*n_signal:(2*n_signal+n_numu)]

    assert(np.abs(np.sum(covariance) - np.sum(rescaled_covariance)) < 1.e-3)

    # compute residuals
    residuals = np.concatenate([observed_nue - (by_hand_mod + predicted_background),
                                (observed_numu - predicted_numu)])
    #print residuals, np.sum(residuals)
    # invert covariance
    inv_cov = np.linalg.inv(covariance)
    #inv_cov = np.linalg.pinv(covariance) # pseudo inverse to check for numerical stability

    # calculate chi^2
    chi2 = np.dot(residuals,np.dot(inv_cov,residuals))

    # fin
    return chi2

def CalculateChi2BackgroundExpectations(predicted_neutrino_background):
    observed_neutrino_nue = mb_nue_analysis_data
    observed_neutrino_numu = mb_numu_analyis_data

    predicted_neutrino_numu = mb_numu_analyis_prediction

    predicted_neutrino_signal = observed_neutrino_nue - predicted_neutrino_background

    Excess =  np.sum(mb_nominal_excess)
    FracExcess = np.abs(np.sum(predicted_neutrino_signal) - Excess)/Excess

    ##Fix the total number of signal events within 5%
    if FracExcess > 0.05 :
        return np.inf                                                          

    return CalculateChi2(fractional_covariance_matrix,
                         predicted_neutrino_signal, mb_nue_analysis_predicted_background, predicted_neutrino_numu,
                         observed_neutrino_nue, observed_neutrino_numu)

    
    #return CalculateChi2(fractional_covariance_matrix,
    #                     predicted_neutrino_signal, predicted_neutrino_background, predicted_neutrino_numu,
    #                     observed_neutrino_nue, observed_neutrino_numu)


def CalculateChi2SignalExpectations(predicted_neutrino_signal):
    observed_neutrino_nue = mb_nue_analysis_data
    observed_neutrino_numu = mb_numu_analyis_data

    predicted_neutrino_background = mb_nue_analysis_predicted_background

    predicted_neutrino_numu = mb_numu_analyis_prediction

    return CalculateChi2(fractional_covariance_matrix ,
                         predicted_neutrino_signal, 
                         predicted_neutrino_background, 
                         predicted_neutrino_numu,
                         observed_neutrino_nue, observed_neutrino_numu)

#################################### END MINIBOONE ANALYSIS ######################################

