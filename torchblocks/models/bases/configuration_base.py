""" Configuration base class and utilities."""
import copy
import json
import logging
import os

logger = logging.getLogger(__name__)
CONFIG_NAME = 'config.json'


class TrainConfig(object):
    def __init__(self, **kwargs):
        self.model_type = kwargs.pop("model_type", '')
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_pretrained(cls, trained_model_path, **kwargs):
        config_dict, kwargs = cls.get_config_dict(trained_model_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(cls, trained_model_path, **kwargs):
        if os.path.isdir(trained_model_path):
            config_file = os.path.join(trained_model_path, CONFIG_NAME)
        elif os.path.isfile(trained_model_path):
            config_file = trained_model_path
        else:
            raise ValueError(f"Can't load config from '{trained_model_path}'")
        config_dict = cls._dict_from_json_file(config_file)
        logger.info("loading configuration file {}".format(config_file))
        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        config = cls(**config_dict)
        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file):
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return "{} {}".format(self.__class__.__name__, self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        if hasattr(self, "model_type"):
            output["model_type"] = self.model_type
        return output

    def to_json_string(self):
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def update(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
