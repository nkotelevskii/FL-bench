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
            y_pred, log_prob, _ = model.train_forward(x)
            logits = y_pred.alpha.log()
            loss += criterion(y_pred, y, log_prob).item()
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
            print(f"***************** logprob: {log_prob.cpu().mean()}")
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
    logprob_threshold: float,
    criterion: torch.nn.Module,
    device=torch.device("cpu"),
) -> tuple[float, float, int]:
    local_model.eval()
    global_model.eval()
    correct_local = 0
    correct_global = 0
    correct_decision = 0

    loss_local = 0
    loss_global = 0
    sample_num = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        y_pred_local, log_prob_local, _ = local_model.train_forward(x)
        y_pred_global, log_prob_global, _ = global_model.train_forward(x)

        logits_local = y_pred_local.alpha.log()
        logits_global = y_pred_global.alpha.log()
        
        threshold_torch = logprob_threshold * torch.ones_like(log_prob_local)

        logit_after_decision = torch.where((log_prob_local > threshold_torch)[..., None], logits_local, logits_global)

        loss_local += criterion(y_pred_local, y, log_prob_local).item()
        loss_global += criterion(y_pred_global, y, log_prob_global).item()

        pred_local = torch.argmax(logits_local, -1)
        pred_global = torch.argmax(logits_global, -1)
        pred_after_decision = torch.argmax(logit_after_decision, -1)

        correct_local += (pred_local == y).sum().item()
        correct_global += (pred_global == y).sum().item()
        correct_decision += (pred_after_decision == y).sum().item()


        sample_num += len(y)
    return correct_decision, loss_local, correct_local, loss_global, correct_global, sample_num


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