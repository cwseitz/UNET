from abc import abstractmethod
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

#import jax.numpy as jnp
import torch
import torch.nn as nn
#from flax import linen
#from numpyro.distributions import Distribution
#from pyro.infer.predictive import Predictive

from .._types import LossRecord

from ._decorators import auto_move_data
#from ._pyro import AutoMoveDataPredictive


class LossRecorder:
    """
    Loss signature for models.
    This class provides an organized way to record the model loss, as well as
    the components of the ELBO. This may also be used in MLE, MAP, EM methods.
    The loss is used for backpropagation during inference. The other parameters
    are used for logging/early stopping during inference.
    Parameters
    ----------
    loss
        Tensor with loss for minibatch. Should be one dimensional with one value.
        Note that loss should be a :class:`~torch.Tensor` and not the result of `.item()`.
    reconstruction_loss
        Reconstruction loss for each observation in the minibatch.
    kl_local
        KL divergence associated with each observation in the minibatch.
    kl_global
        Global kl divergence term. Should be one dimensional with one value.
    **kwargs
        Additional metrics can be passed as keyword arguments and will
        be available as attributes of the object.
    """

    def __init__(
        self,
        loss: LossRecord,
        reconstruction_loss: Optional[LossRecord] = None,
        kl_local: Optional[LossRecord] = None,
        kl_global: Optional[LossRecord] = None,
        **kwargs,
    ):

        default = (
            torch.tensor(0.0) if isinstance(loss, torch.Tensor) else jnp.array(0.0)
        )
        if reconstruction_loss is None:
            reconstruction_loss = default
        if kl_local is None:
            kl_local = default
        if kl_global is None:
            kl_global = default

        self._loss = loss if isinstance(loss, dict) else dict(loss=loss)
        self._reconstruction_loss = (
            reconstruction_loss
            if isinstance(reconstruction_loss, dict)
            else dict(reconstruction_loss=reconstruction_loss)
        )
        self._kl_local = (
            kl_local if isinstance(kl_local, dict) else dict(kl_local=kl_local)
        )
        self._kl_global = (
            kl_global if isinstance(kl_global, dict) else dict(kl_global=kl_global)
        )
        self.extra_metric_attrs = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.extra_metric_attrs.append(key)

    @staticmethod
    def _get_dict_sum(dictionary):
        total = 0.0
        for value in dictionary.values():
            total += value
        return total

#    @property
#    def loss(self) -> Union[torch.Tensor, jnp.ndarray]:
#        return self._get_dict_sum(self._loss)

#    @property
#    def reconstruction_loss(self) -> Union[torch.Tensor, jnp.ndarray]:
#        return self._get_dict_sum(self._reconstruction_loss)

#    @property
#    def kl_local(self) -> Union[torch.Tensor, jnp.ndarray]:
#        return self._get_dict_sum(self._kl_local)

#    @property
#    def kl_global(self) -> Union[torch.Tensor, jnp.ndarray]:
#        return self._get_dict_sum(self._kl_global)


class BaseModuleClass(nn.Module):
    """Abstract class for scvi-tools modules."""

    def __init__(
        self,
    ):
        super().__init__()

    @property
    def device(self):
        device = list(set(p.device for p in self.parameters()))
        if len(device) > 1:
            raise RuntimeError("Module tensors on multiple devices.")
        return device[0]

    @auto_move_data
    def forward(
        self,
        tensors,
        get_inference_input_kwargs: Optional[dict] = None,
        get_generative_input_kwargs: Optional[dict] = None,
        inference_kwargs: Optional[dict] = None,
        generative_kwargs: Optional[dict] = None,
        loss_kwargs: Optional[dict] = None,
        compute_loss=True,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, LossRecorder],
    ]:
        """
        Forward pass through the network.
        Parameters
        ----------
        tensors
            tensors to pass through
        get_inference_input_kwargs
            Keyword args for `_get_inference_input()`
        get_generative_input_kwargs
            Keyword args for `_get_generative_input()`
        inference_kwargs
            Keyword args for `inference()`
        generative_kwargs
            Keyword args for `generative()`
        loss_kwargs
            Keyword args for `loss()`
        compute_loss
            Whether to compute loss on forward pass. This adds
            another return value.
        """
        return _generic_forward(
            self,
            tensors,
            inference_kwargs,
            generative_kwargs,
            loss_kwargs,
            get_inference_input_kwargs,
            get_generative_input_kwargs,
            compute_loss,
        )

    @abstractmethod
    def _get_inference_input(self, tensors: Dict[str, torch.Tensor], **kwargs):
        """Parse tensors dictionary for inference related values."""

    @abstractmethod
    def _get_generative_input(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor],
        **kwargs,
    ):
        """Parse tensors dictionary for generative related values."""

    @abstractmethod
    def inference(
        self,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, torch.distributions.Distribution]]:
        """
        Run the inference (recognition) model.
        In the case of variational inference, this function will perform steps related to
        computing variational distribution parameters. In a VAE, this will involve running
        data through encoder networks.
        This function should return a dictionary with str keys and :class:`~torch.Tensor` values.
        """
        pass

    @abstractmethod
    def generative(
        self, *args, **kwargs
    ) -> Dict[str, Union[torch.Tensor, torch.distributions.Distribution]]:
        """
        Run the generative model.
        This function should return the parameters associated with the likelihood of the data.
        This is typically written as :math:`p(x|z)`.
        This function should return a dictionary with str keys and :class:`~torch.Tensor` values.
        """
        pass

    @abstractmethod
    def loss(self, *args, **kwargs) -> LossRecorder:
        """
        Compute the loss for a minibatch of data.
        This function uses the outputs of the inference and generative functions to compute
        a loss. This many optionally include other penalty terms, which should be computed here.
        This function should return an object of type :class:`~scvi.module.base.LossRecorder`.
        """
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        """Generate samples from the learned model."""
        pass

def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param

def _generic_forward(
    module,
    tensors,
    inference_kwargs,
    generative_kwargs,
    loss_kwargs,
    get_inference_input_kwargs,
    get_generative_input_kwargs,
    compute_loss,
):
    """Core of the forward call shared by PyTorch- and Jax-based modules."""
    inference_kwargs = _get_dict_if_none(inference_kwargs)
    generative_kwargs = _get_dict_if_none(generative_kwargs)
    loss_kwargs = _get_dict_if_none(loss_kwargs)
    get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)
    get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)

    inference_inputs = module._get_inference_input(
        tensors, **get_inference_input_kwargs
    )
    inference_outputs = module.inference(**inference_inputs, **inference_kwargs)
    generative_inputs = module._get_generative_input(
        tensors, inference_outputs, **get_generative_input_kwargs
    )
    generative_outputs = module.generative(**generative_inputs, **generative_kwargs)
    if compute_loss:
        losses = module.loss(
            tensors, inference_outputs, generative_outputs, **loss_kwargs
        )
        return inference_outputs, generative_outputs, losses
    else:
        return inference_outputs, generative_outputs
