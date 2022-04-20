import inspect
import logging
import os
import warnings
from uuid import uuid4

import numpy as np
import torch

class BaseModel:
    """Abstract class for models."""

    def __init__(self, data=None):

        self.is_trained_ = False
        self._model_summary_string = ""
        self.train_indices_ = None
        self.test_indices_ = None
        self.validation_indices_ = None
        self.history_ = None
        self._data_loader_cls = AnnDataLoader


    def to_device(self, device: Union[str, int]):
    
        """
        Move model to device.

        Parameters
        ----------
        device
            Device to move model to. Options: 'cpu' for CPU, integer GPU index (eg. 0),
            or 'cuda:X' where X is the GPU index (eg. 'cuda:0'). See torch.device for more info.

        Examples
        --------
        >>> adata = arwn.data.synthetic_iid()
        >>> model = arwn.model.arwn(adata)
        >>> model.to_device('cpu')      # moves model to CPU
        >>> model.to_device('cuda:0')   # moves model to GPU 0
        >>> model.to_device(0)          # also moves model to GPU 0
        """
        my_device = torch.device(device)
        self.module.to(my_device)

    @property
    def device(self) -> str:
        """The current device that the module's params are on."""
        return self.module.device

    def _check_if_trained(
        self, warn: bool = True, message: str = _UNTRAINED_WARNING_MESSAGE
    ):
        """
        Check if the model is trained.

        If not trained and `warn` is True, raise a warning, else raise a RuntimeError.
        """
        if not self.is_trained_:
            if warn:
                warnings.warn(message)
            else:
                raise RuntimeError(message)

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self.is_trained_

    @is_trained.setter
    def is_trained(self, value):
        self.is_trained_ = value

    def _get_user_attributes(self):
        """Returns all the self attributes defined in a model class, e.g., self.is_trained_."""
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [
            a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))
        ]
        attributes = [a for a in attributes if not a[0].startswith("_abc_")]
        return attributes


