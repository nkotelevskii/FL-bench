from typing import Optional, Union

import torch
from pyro.distributions import TransformModule
from pyro.distributions.transforms import AffineCoupling, Permute, ComposeTransformModule, Radial
from pyro.nn import DenseNN
import math

class PermuteModule(torch.nn.Module):
    def __init__(self, permutation):
        super(PermuteModule, self).__init__()
        self.permute = Permute(permutation)

    def forward(self, x):
        return self.permute(x)

    def inverse(self, y):
        return self.permute.inv(y)


def initialize_realnvp_flow(input_dim: int = 2, split_dim: int = 1, n_transforms: int = 2,
                            device: Optional[Union[str, torch.device]] = 'cpu') -> ComposeTransformModule:
    """
    The function creates stack of RealNVP normalizing flows. It takes reordering into account.
    :param input_dim:
    :param split_dim:
    :param n_transforms:
    :param device:
    :return:
    """
    param_dims = [input_dim - split_dim, input_dim - split_dim]
    hidden_multiplier = 2
    hypernet = DenseNN(
        split_dim, [hidden_multiplier * input_dim], param_dims, nonlinearity=torch.nn.ELU()).to(device)
    list_of_transforms: list[TransformModule] = [
        AffineCoupling(split_dim, hypernet).to(device)]
    for i in range(n_transforms):
        order = ((i % 2) * 2 - 1)
        hypernet = DenseNN(split_dim, [hidden_multiplier * input_dim], param_dims).to(device)
        list_of_transforms.extend([
            # Permute(permutation=torch.LongTensor(
            #     [i for i in range(input_dim)][::order]).to(device)),
            # PermuteModule(permutation=torch.LongTensor(
            # [i for i in range(input_dim)][::order]).to(device)),
            AffineCoupling(split_dim, hypernet).to(device),
        ])
    composed_list = ComposeTransformModule(list_of_transforms)
    return composed_list


def initialize_radial_flow(input_dim: int = 2, n_transforms: int = 2,
                           device: Optional[Union[str, torch.device]] = 'cpu') -> ComposeTransformModule:
    """
    The function creates stack of Radial flows for a given number of transformations
    :param input_dim:
    :param n_transforms:
    :param device:
    :return:
    """
    list_of_transforms = []
    for i in range(n_transforms):
        list_of_transforms.append(Radial(input_dim=input_dim).to(device))
    composed_list = ComposeTransformModule(list_of_transforms)
    return composed_list


def process_flow_batch(
        local_flow: ComposeTransformModule,
        batch_embeddings: torch.Tensor) -> torch.Tensor:

    neutralized_embeddings = local_flow(batch_embeddings)

    # Compute log-probability
    const = neutralized_embeddings.shape[1] * math.log(2 * math.pi)
    norm = torch.einsum("...ij,...ij->...i",
                        neutralized_embeddings, neutralized_embeddings)
    normal_log_prob = -0.5 * (const + norm)

    log_prob = normal_log_prob + local_flow.log_abs_det_jacobian(
        batch_embeddings, neutralized_embeddings)
    return log_prob
