from scipy.special import digamma
import numpy as np


def mutual_information(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)
    mi = -1. * np.sum(alpha / alpha_0 * (np.log(alpha / alpha_0) - digamma(alpha + 1) + digamma(alpha_0 + 1)),
                      axis=1)

    return mi


def reverse_mutual_information(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)
    rmi = np.sum(alpha / alpha_0 * (np.log(alpha / alpha_0) -
                 digamma(alpha) + digamma(alpha_0)), axis=1)

    return rmi


def expected_pairwise_kullback_leibler(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)
    epkl = np.sum(alpha / alpha_0 * (digamma(alpha + 1) - digamma(alpha_0 + 1) - digamma(alpha) + digamma(alpha_0)),
                  axis=1)

    return epkl


def expected_entropy(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)
    exp_entropy = np.sum(alpha / alpha_0 * (digamma(alpha + 1) - digamma(alpha_0 + 1)),
                         axis=1)
    return exp_entropy
