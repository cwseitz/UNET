from typing import Dict, Type, Union

#import jax.numpy as jnp
import torch

#from arwn.data.fields import BaseAnnDataField

Number = Union[int, float]
#AnnDataField = Type[BaseAnnDataField]
#Tensor = Union[torch.Tensor, jnp.ndarray]
Tensor = torch.Tensor
LossRecord = Union[Dict[str, Tensor], Tensor]
