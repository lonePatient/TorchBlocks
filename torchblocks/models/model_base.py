import logging
import os
import torch
from torch import nn
import torch.nn.init as init
from .configuration_base import TrainConfig

logger = logging.getLogger(__name__)
WEIGHTS_NAME = 'pytorch_model.bin'

INIT2FCT = {'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_normal': init.kaiming_normal_,
            'kaiming_uniform': init.kaiming_uniform_,
            'orthogonal': init.orthogonal_,
            'sparse': init.sparse_,
            'normal': init.normal_,
            'uniform': init.uniform_
            }

class TrainModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, TrainConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `TrainConfig`. "
                "To create a model from a train model use "
                "`model = {}.load(model_path)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

            Arguments:
                save_directory: directory to which to save.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        if hasattr(model_to_save.config, 'save_pretrained'):
            model_to_save.config.save_pretrained(save_directory)
        else:
            raise ValueError("Make sure that:\n 'config' is a correct config file")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    @classmethod
    def from_pretrained(cls, model_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        # Load config if we don't provide a configuration
        if not isinstance(config, TrainConfig):
            config_path = config if config is not None else model_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                **kwargs)
        else:
            model_kwargs = kwargs
        # Load model
        if os.path.isdir(model_path):
            if os.path.isfile(os.path.join(model_path, WEIGHTS_NAME)):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(model_path, WEIGHTS_NAME)
            else:
                logger.warning(
                    "Error no file named {} found in directory {} ".format(
                        [WEIGHTS_NAME, ], model_path))
                archive_file = None
        elif os.path.isfile(model_path):
            archive_file = model_path
        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)
        if state_dict is None:
            try:
                state_dict = torch.load(archive_file, map_location="cpu")
            except Exception:
                logger.warning(
                    "Unable to load weights from pytorch checkpoint file. ")
        if state_dict:
            model.load_state_dict(state_dict)
        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        return model

    def _init_weights(self, m, initial_method=None):
        r"""A method used to initialize the weights of PyTorch models.
        :param net: a PyTorch model
        :param str initial_method: one of the following initializations.
                - xavier_uniform
                - xavier_normal (default)
                - kaiming_normal, or msra
                - kaiming_uniform
                - orthogonal
                - sparse
                - normal
                - uniform
        """
        init_method = INIT2FCT[initial_method]
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.kaiming_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.Embedding):
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                if hasattr(self.config, 'initializer_range'):
                    initializer_range = self.config.initializer_range
                else:
                    initializer_range = 0.02
                m.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif m is not None and hasattr(m, 'weight') and \
                hasattr(m.weight, "requires_grad"):
            init_method(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
