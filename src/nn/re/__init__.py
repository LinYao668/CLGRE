from .mymodel import get_auto_mymodel_re_model, get_mymodel_re_model_config
from ...registry import BaseParent


class AutoReTaskModelConfig(BaseParent):

    registry = {}

    @classmethod
    def create(cls, class_key, labels, **kwargs):
        return cls.registry[class_key](labels, **kwargs)


class AutoReTaskModel(BaseParent):

    registry = {}

    @classmethod
    def create(cls, class_key, model_type="bert", **kwargs):
        return cls.registry[class_key](model_type, **kwargs)


AutoReTaskModelConfig.add_to_registry("mymodel", get_mymodel_re_model_config)

AutoReTaskModel.add_to_registry("mymodel", get_auto_mymodel_re_model)
