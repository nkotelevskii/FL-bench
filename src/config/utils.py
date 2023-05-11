import os
from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import random
import numpy as np
from path import Path
from torch.utils.data import DataLoader
from src.config.nat_pn.loss import BayesianLoss, LogMarginalLoss
from src.config.models import NatPnModel
from src.config.uncertainty_metrics import (
    mutual_information,
    reverse_mutual_information,
    expected_entropy,
    expected_pairwise_kullback_leibler,
    load_dataloaders,
    load_dataset,
    load_model
)
from copy import deepcopy
import math

_PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
OUT_DIR = _PROJECT_DIR / "out"
TEMP_DIR = _PROJECT_DIR / "temp"


def fix_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clone_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )


def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module], requires_name=False
) -> Union[List[torch.Tensor], Tuple[List[str], List[torch.Tensor]]]:
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)

    if requires_name:
        return keys, parameters
    else:
        return parameters


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device=torch.device("cpu"),
) -> Tuple[float, float, int]:
    model.eval()
    correct = 0
    loss = 0
    sample_num = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        if isinstance(criterion, BayesianLoss) or isinstance(criterion, LogMarginalLoss):
            y_pred, log_prob, embeddings = model.train_forward(x)
            logits = y_pred.alpha.log()
            loss += criterion(y_pred, y, log_prob, embeddings).item()
        else:
            logits = model(x)
            loss += criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        correct += (pred == y).sum().item()
        sample_num += len(y)
    return loss, correct, sample_num


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device=torch.device("cpu"),
) -> Tuple[float, float, int]:
    model.eval()
    correct = 0
    sample_num = 0
    mean_log_prob = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, NatPnModel):
            y_pred, log_prob, _ = model.train_forward(x)
            # print(f"***************** logprob: {log_prob.cpu().mean()}")
            logits = y_pred.alpha
            mean_log_prob.append(log_prob.cpu().numpy())
        else:
            logits = model(x)
        pred = torch.argmax(logits, -1)
        correct += (pred == y).sum().item()
        sample_num += len(y)

    return correct / sample_num, np.mean(np.hstack(mean_log_prob))


@torch.no_grad()
def evaluate_switch(
    local_model: torch.nn.Module,
    global_model: torch.nn.Module,
    dataloader: DataLoader,
    threshold: float,
    uncertainty_measure: str,
    device=torch.device("cpu"),
) -> tuple[float, float, int]:
    local_model.eval()
    global_model.eval()
    correct_local = 0
    correct_global = 0
    correct_decision = 0

    sample_num = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        y_pred_local, log_prob_local, _ = local_model.train_forward(x, clamp=False)
        y_pred_global, _, _ = global_model.train_forward(x, clamp=False)

        alpha_local = y_pred_local.alpha
        alpha_global = y_pred_global.alpha

        threshold_torch = threshold * torch.ones_like(alpha_local[:, 0])

        if uncertainty_measure == 'mi':
            local_measure = mutual_information(alpha=alpha_local.cpu().numpy())
        elif uncertainty_measure == 'rmi':
            local_measure = reverse_mutual_information(
                alpha=alpha_local.cpu().numpy())
        elif uncertainty_measure == 'epkl':
            local_measure = expected_pairwise_kullback_leibler(
                alpha=alpha_local.cpu().numpy())
        elif uncertainty_measure == 'entropy':
            local_measure = y_pred_local.entropy().cpu().numpy()
        elif uncertainty_measure == 'log_prob':
            local_measure = log_prob_local.cpu().numpy()
        elif uncertainty_measure == 'categorical_entropy':
            local_measure = expected_entropy(alpha=alpha_local.cpu().numpy())
        else:
            raise ValueError(
                f'No such uncertainty measure available! {uncertainty_measure}')

        local_measure = torch.tensor(local_measure, device=device) * torch.ones_like(alpha_local[:, 0])
        if uncertainty_measure != 'log_prob':
            alphas_after_decision = torch.where((local_measure < threshold_torch)[..., None],
                                                alpha_local, alpha_global)
        else:
            alphas_after_decision = torch.where((local_measure > threshold_torch)[..., None],
                                    alpha_local, alpha_global)

        pred_local = torch.argmax(alpha_local, -1)
        pred_global = torch.argmax(alpha_global, -1)
        pred_after_decision = torch.argmax(alphas_after_decision, -1)

        correct_local += (pred_local == y).sum().item()
        correct_global += (pred_global == y).sum().item()
        correct_decision += (pred_after_decision == y).sum().item()

        sample_num += len(y)
    return correct_decision, correct_local, correct_global, sample_num


@torch.no_grad()
def validate_accuracy_per_client(
        dataset_name: str,
        backbone: str,
        stopgrad: bool,
        criterion: torch.nn.Module,
        all_params_dict: dict[int, torch.Tensor],
        device: str,
        validate_only_classifier: bool = False,
) -> None:
    data_indices, trainset, testset = load_dataset(dataset_name=dataset_name)
    for index in ['global'] + [i for i in range(len(data_indices))]:
        print("model index is: ", index)

        current_model = load_model(
            dataset_name=dataset_name,
            backbone=backbone,
            stopgrad=stopgrad,
            index=index,
            all_params_dict=all_params_dict,
        )

        for dataset_index in range(len(data_indices)):
            _, testloader, _ = load_dataloaders(
                client_id=dataset_index, data_indices=data_indices, trainset=trainset, testset=testset
            )
            test_loss, test_correct, test_sample_num = evaluate(
                model=current_model,
                dataloader=testloader,
                criterion=criterion
            )
            if isinstance(current_model, NatPnModel) and validate_only_classifier:
                correct, overall = evaluate_only_classifier(
                    model=current_model, dataloader=testloader, device=device)
                print(
                    f"{dataset_index}: {test_correct / test_sample_num} \t Accuracy only classifier: {correct.sum() / overall}")
            else:
                print(f"{dataset_index}: {test_correct / test_sample_num}")


@torch.no_grad()
def evaluate_only_classifier(
    model: NatPnModel,
    dataloader: DataLoader,
    device,
):
    all_correct = []
    n_samples = 0
    model.eval()
    for x, y in dataloader:
        x = x.to(device)
        pred_raw = model.classifier(model.base(x))
        pred_logit = pred_raw.logits
        pred = pred_logit.argmax(-1).cpu().numpy()
        all_correct.append(pred == y.cpu().numpy())
        n_samples += len(y.cpu().numpy())
    return np.hstack(all_correct), n_samples


@torch.no_grad()
def reset_flow_and_classifier_params(state_dict: OrderedDict):
    new_state_dict = deepcopy(state_dict)
    for k, v in state_dict.items():
        if not (k.startswith("flow") or k.startswith("classifier")):
            new_state_dict[k] = v
        else:
            new_state_dict[k] = torch.nn.Parameter(torch.nn.init.kaiming_uniform(
                new_state_dict[k][None], a=math.sqrt(5))[0])
    return new_state_dict
