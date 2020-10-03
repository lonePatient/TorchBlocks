import json
import copy


class InputExample:
    """
    A single training/test example for simple sequence classification.
        texts: 列表形式，比如 [text_a,text_b]
        label: 标签信息,
        label_ids: 标签列表，比如多标签，NER等任务
    """
    def __init__(self,
                 guid,
                 texts,
                 label=None,
                 label_ids=None,
                 **kwargs):
        self.guid = guid
        self.texts = texts
        self.label = label
        self.label_ids = label_ids

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures:
    """
    A single set of features of processor.
    Property names are the same names as the corresponding inputs to a model.
    """

    def __init__(self,
                 input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 label=None,
                 label_ids=None,
                 **kwargs):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.label_ids = label_ids
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
