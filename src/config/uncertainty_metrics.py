from scipy.special import digamma
import numpy as np
from torch.utils.data import DataLoader, Subset
import pickle
from torchvision.transforms import Compose, Normalize
from data.utils.constants import MEAN, STD
from data.utils.datasets import DATASETS
import torch
from src.config.models import NatPnModel
from copy import deepcopy
from typing import Optional


def H(n):
    """Returns an approximate value of n-th harmonic number.

       http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + np.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)


def load_dataset(dataset_name: str,
                  normalization_name: Optional[str] = None) -> tuple[list[list[int]], Subset, Subset]:
    if normalization_name is None:
        normalization_name = dataset_name
    with open(f'../data/{dataset_name}/partition.pkl', 'rb') as file:
        partition = pickle.load(file, )
    data_indices: list[list[int]] = partition["data_indices"]
    transform = Compose(
        [Normalize(MEAN[normalization_name], STD[normalization_name])]
    )

    dataset = DATASETS[dataset_name](
        root=f"../data/{dataset_name}",
        args=None,
        transform=transform,
        target_transform=None,
    )

    trainset: Subset = Subset(dataset, indices=[])
    testset: Subset = Subset(dataset, indices=[])

    return data_indices, trainset, testset


def load_dataloaders(
        client_id: int,
        data_indices: list[list[int]],
        trainset: Subset,
        testset: Subset
) -> tuple[DataLoader, DataLoader, DataLoader]:
    trainset.indices = data_indices[client_id]["train"]
    trainloader = DataLoader(trainset, 32)

    n_val = int(0.6 * len(data_indices[client_id]["test"]))
    np.random.seed(42)
    val_indices = np.random.choice(
        data_indices[client_id]["test"], size=n_val, replace=False, )
    cal_indices = np.array(
        [ind for ind in data_indices[client_id]["test"] if ind not in val_indices])

    testset.indices = val_indices
    testloader = DataLoader(testset, 32, shuffle=False)

    testset.indices = cal_indices
    calloader = DataLoader(testset, 32, shuffle=False)

    return trainloader, testloader, calloader


def load_model(
        dataset_name: str,
        backbone: str,
        stopgrad: bool,
        index: int,
        all_params_dict: dict[int, torch.Tensor],
) -> NatPnModel:
    model = NatPnModel(dataset=dataset_name,
                       backbone=backbone,
                       stop_grad_logp=stopgrad,
                       stop_grad_embeddings=stopgrad,
                       )

    model.load_state_dict(all_params_dict[index], strict=False)
    model.labels = all_params_dict[index]['labels']
    model.labels_frequency = all_params_dict[index]['labels_frequency']
    current_model = deepcopy(model)
    current_model.eval()
    return current_model


@torch.no_grad()
def choose_threshold(
        model: NatPnModel,
        calloader: DataLoader,
        device: str,
        alpha: float = 0.975,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    mis = []  # Mutual Informations
    rmis = []  # Reverse Mutual Informations
    epkls = []  # Expected Pairwise Kullback Leiblers divergences
    entropies = []  # Entropies of Dirichlet
    categorical_entropies = [] # Expected entropy (expectation over Dirichlet of Categoricals)
    log_probs = []  # Logarithms of embedding's density
    for x, y in calloader:
        x, y = x.to(device), y.to(device)
        y_pred, log_prob, local_embeddings = model.train_forward(x, clamp=False)
        alphas = y_pred.alpha.cpu().numpy()

        mi = mutual_information(alpha=alphas)
        rmi = reverse_mutual_information(alpha=alphas)
        epkl = expected_pairwise_kullback_leibler(alpha=alphas)
        entropy = y_pred.entropy().cpu().numpy()
        categorical_entropy = expected_entropy(alpha=alphas)

        mis.append(mi)
        rmis.append(rmi)
        epkls.append(epkl)
        entropies.append(entropy)
        log_probs.append(log_prob.cpu().numpy())
        categorical_entropies.append(categorical_entropy)

    mis = np.hstack(mis)
    rmis = np.hstack(rmis)
    epkls = np.hstack(epkls)
    entropies = np.hstack(entropies)
    log_probs = np.hstack(log_probs)
    categorical_entropies = np.hstack(categorical_entropies)

    threshold_mi = np.quantile(mis, alpha)
    threshold_rmi = np.quantile(rmis, alpha)
    threshold_epkl = np.quantile(epkls, alpha)
    threshold_entropy = np.quantile(entropies, alpha)
    threshold_categorical_entropy = np.quantile(categorical_entropies, alpha)
    threshold_log_prob = np.quantile(log_probs, 1 - alpha)
    thresholds = {
        "mi": threshold_mi,
        "rmi": threshold_rmi,
        "epkl": threshold_epkl,
        "entropy": threshold_entropy,
        "log_prob": threshold_log_prob,
        "categorical_entropy": threshold_categorical_entropy,
    }
    values = {
        "mi": mis,
        "rmi": rmis,
        "epkl": epkls,
        "entropy": entropies,
        "log_prob": log_probs,
        "categorical_entropy": categorical_entropies,
    }
    return thresholds, values

def mutual_information(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)

    # exact
    exact = -1. * np.sum(alpha / alpha_0 * (np.log(alpha / alpha_0) - digamma(alpha + 1) + digamma(alpha_0 + 1)),
                      axis=1)
    # approximate
    diff = 1 - alpha.shape[1] + (1 / (1 + alpha_0[:, 0])) - np.sum(1 / (alpha + 1), axis=1)
    approx = -1. * diff / (2 * alpha_0[:, 0])

    return np.where(alpha_0[:, 0] >= 10000, approx, exact)


def reverse_mutual_information(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)
    exact = np.sum(alpha / alpha_0 * (np.log(alpha / alpha_0) -
                 digamma(alpha) + digamma(alpha_0)), axis=1)

    approx = (alpha.shape[1] - 1) / (2 * alpha_0[:, 0])

    return np.where(alpha_0[:, 0] >= 10000, approx, exact)


def expected_pairwise_kullback_leibler(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)
    exact = np.sum(alpha / alpha_0 * (digamma(alpha + 1) - digamma(alpha_0 + 1) - digamma(alpha) + digamma(alpha_0)),
                  axis=1)

    diff = 2 * alpha.shape[1] - 2 - (1. / (1 + alpha_0[:, 0])) + np.sum(1 / (alpha + 1), axis=1)
    approx = diff / (2 * alpha_0[:, 0])

    return np.where(alpha_0[:, 0] >= 10000, approx, exact)


def expected_entropy(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)
    # exact
    exact = np.sum(alpha / alpha_0 * (digamma(alpha + 1) - digamma(alpha_0 + 1)),
                         axis=1)

    # approximation
    approx = np.sum(alpha / alpha_0 * (np.log(alpha) - np.log(alpha_0)), axis=-1)

    return np.where(alpha_0[:, 0] >= 10000, approx, exact)
